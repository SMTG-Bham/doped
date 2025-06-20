"""
Tests for the ``doped.analysis`` module, which also implicitly tests most of
the ``doped.utils.parsing`` module, and some ``doped.thermodynamics``
functions.
"""

import gzip
import os
import shutil
import unittest
import warnings
from unittest.mock import patch

import matplotlib as mpl
import numpy as np
import pytest
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.defects.core import DefectType
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.dos import FermiDos
from test_thermodynamics import custom_mpl_image_compare

from doped.analysis import (
    DefectParser,
    DefectsParser,
    defect_entry_from_paths,
    defect_from_structures,
    defect_name_from_structures,
)
from doped.core import _orientational_degeneracy_warning
from doped.generation import DefectsGenerator, get_defect_name_from_defect, get_defect_name_from_entry
from doped.utils.eigenvalues import get_eigenvalue_analysis
from doped.utils.parsing import (
    Vasprun,
    _num_electrons_from_charge_state,
    _simple_spin_degeneracy_from_num_electrons,
    get_defect_type_and_composition_diff,
    get_defect_type_site_idxs_and_unrelaxed_structure,
    get_magnetization_from_vasprun,
    get_outcar,
    get_procar,
    get_vasprun,
    spin_degeneracy_from_vasprun,
)
from doped.utils.symmetry import (
    get_orientational_degeneracy,
    point_symmetry_from_defect_entry,
    point_symmetry_from_structure,
)

mpl.use("Agg")  # don't show interactive plots if testing from CLI locally


def if_present_rm(path):
    """
    Remove file or directory if it exists.
    """
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def _create_dp_and_capture_warnings(*args, **kwargs):
    with warnings.catch_warnings(record=True) as w:
        try:
            dp = DefectsParser(*args, **kwargs)
        except Exception as e:
            print([warn.message for warn in w])  # for debugging
            raise e
    print([warn.message for warn in w])  # for debugging
    return dp, w


def _remove_metadata_keys_from_dict(d: dict) -> dict:
    # recursively pop all keys with "@" in them (i.e. remove version/class metadata which might change)
    for key in list(d.keys()):
        if "@" in key:
            d.pop(key)
        elif isinstance(d[key], list):
            for i, item in enumerate(d[key]):
                if isinstance(item, dict):
                    d[key][i] = _remove_metadata_keys_from_dict(item)
        elif isinstance(d[key], dict):
            d[key] = _remove_metadata_keys_from_dict(d[key])

    return d


class DefectsParsingTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module_path = os.path.dirname(os.path.abspath(__file__))
        cls.EXAMPLE_DIR = os.path.join(cls.module_path, "../examples")
        cls.CdTe_EXAMPLE_DIR = os.path.abspath(os.path.join(cls.module_path, "../examples/CdTe"))
        cls.v_Cd_example_dir = os.path.join(cls.CdTe_EXAMPLE_DIR, "v_Cd_example_data")

        cls.moved_v_Cd_example_dirs = []

        for i in os.listdir(cls.v_Cd_example_dir):
            # first clear these directories from higher level CdTe example folder, in case previous test
            # failed without clearing the directories:
            if_present_rm(os.path.join(cls.CdTe_EXAMPLE_DIR, i))
            shutil.move(os.path.join(cls.v_Cd_example_dir, i), os.path.join(cls.CdTe_EXAMPLE_DIR, i))
            cls.moved_v_Cd_example_dirs.append(i)

    def setUp(self):
        self.CdTe_BULK_DATA_DIR = os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_bulk/vasp_ncl")
        self.CdTe_dielectric = np.array([[9.13, 0, 0], [0.0, 9.13, 0], [0, 0, 9.13]])  # CdTe
        self.CdTe_chempots = loadfn(os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_chempots.json"))

        self.YTOS_EXAMPLE_DIR = os.path.join(self.module_path, "../examples/YTOS")
        self.ytos_dielectric = [  # from legacy Materials Project
            [40.71948719643814, -9.282128210266565e-14, 1.26076160303219e-14],
            [-9.301652644020242e-14, 40.71948719776858, 4.149879443489052e-14],
            [5.311743673463141e-15, 2.041077680836527e-14, 25.237620491130023],
        ]

        self.Sb2Se3_DATA_DIR = os.path.join(self.module_path, "data/Sb2Se3")
        self.Sb2Se3_dielectric = np.array([[85.64, 0, 0], [0.0, 128.18, 0], [0, 0, 15.00]])

        self.Sb2Si2Te6_dielectric = [44.12, 44.12, 17.82]
        self.Sb2Si2Te6_EXAMPLE_DIR = os.path.join(self.EXAMPLE_DIR, "Sb2Si2Te6")

        self.V2O5_DATA_DIR = os.path.join(self.module_path, "data/V2O5")
        self.SrTiO3_DATA_DIR = os.path.join(self.module_path, "data/SrTiO3")
        self.ZnS_DATA_DIR = os.path.join(self.module_path, "data/ZnS")
        self.SOLID_SOLUTION_DATA_DIR = os.path.join(self.module_path, "data/solid_solution")
        self.CaO_DATA_DIR = os.path.join(self.module_path, "data/CaO")
        self.BiOI_DATA_DIR = os.path.join(self.module_path, "data/BiOI")
        self.shallow_O_Se_DATA_DIR = os.path.join(self.module_path, "data/shallow_O_Se_+1")
        self.Se_dielectric = np.array([0.627551, 0.627551, 0.943432]) + np.array(
            [6.714217, 6.714317, 10.276149]
        )

        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.prim_cdte = Structure.from_file(f"{self.CdTe_EXAMPLE_DIR}/relaxed_primitive_POSCAR")
        self.ytos_bulk_supercell = Structure.from_file(f"{self.EXAMPLE_DIR}/YTOS/Bulk/POSCAR")
        self.lmno_primitive = Structure.from_file(f"{self.data_dir}/Li2Mn3NiO8_POSCAR")
        self.non_diagonal_ZnS = Structure.from_file(f"{self.data_dir}/non_diagonal_ZnS_supercell_POSCAR")
        self.Cu2SiSe3_EXAMPLE_DIR = os.path.join(self.module_path, "../examples/Cu2SiSe3")
        self.MgO_EXAMPLE_DIR = os.path.join(self.module_path, "../examples/MgO")

    @classmethod
    def tearDownClass(cls):
        for i in ["CdTe_bulk", "v_Cd_0", "v_Cd_-1", "v_Cd_-2"]:
            shutil.move(os.path.join(cls.CdTe_EXAMPLE_DIR, i), os.path.join(cls.v_Cd_example_dir, i))
            if_present_rm(os.path.join(cls.CdTe_EXAMPLE_DIR, i))

    def tearDown(self):
        if_present_rm(os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_defect_dict.json.gz"))
        if_present_rm(os.path.join(self.data_dir, "Magnetization_Tests/CdTe/CdTe_defect_dict.json.gz"))
        if_present_rm(os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_test_defect_dict.json"))
        if_present_rm(os.path.join(self.CdTe_EXAMPLE_DIR, "test_pop.json"))
        if_present_rm(os.path.join(self.YTOS_EXAMPLE_DIR, "Y2Ti2S2O5_defect_dict.json.gz"))
        if_present_rm(os.path.join(self.Sb2Si2Te6_EXAMPLE_DIR, "SiSbTe3_defect_dict.json.gz"))
        if_present_rm(os.path.join(self.Sb2Se3_DATA_DIR, "defect/Sb2Se3_defect_dict.json.gz"))
        if_present_rm("V2O5_test")
        if_present_rm(os.path.join(self.SrTiO3_DATA_DIR, "SrTiO3_defect_dict.json.gz"))
        if_present_rm(os.path.join(self.ZnS_DATA_DIR, "ZnS_defect_dict.json.gz"))
        if_present_rm(os.path.join(self.CaO_DATA_DIR, "CaO_defect_dict.json.gz"))
        if_present_rm(os.path.join(self.BiOI_DATA_DIR, "BiOI_defect_dict.json.gz"))
        if_present_rm(os.path.join(self.shallow_O_Se_DATA_DIR, "Se_defect_dict.json.gz"))

        for backup_file in ["OUTCAR.gz", "vasprun.xml.gz"]:
            if os.path.exists(f"{self.shallow_O_Se_DATA_DIR}/sub_1_O_on_Se_1/vasp_std/{backup_file}.bak"):
                shutil.move(
                    f"{self.shallow_O_Se_DATA_DIR}/sub_1_O_on_Se_1/vasp_std/{backup_file}.bak",
                    f"{self.shallow_O_Se_DATA_DIR}/sub_1_O_on_Se_1/vasp_std/{backup_file}",
                )
        if_present_rm(f"{self.shallow_O_Se_DATA_DIR}/sub_1_O_on_Se_1/vasp_std/OUTCAR")
        if_present_rm(f"{self.shallow_O_Se_DATA_DIR}/sub_1_O_on_Se_1/vasp_std/vasprun.xml")

        for i in os.listdir(self.SOLID_SOLUTION_DATA_DIR):
            if "json" in i:
                if_present_rm(os.path.join(self.SOLID_SOLUTION_DATA_DIR, i))

        for i in os.listdir(f"{self.YTOS_EXAMPLE_DIR}/Bulk"):
            if i.startswith("."):
                if_present_rm(f"{self.YTOS_EXAMPLE_DIR}/Bulk/{i}")

        for i in os.listdir(f"{self.YTOS_EXAMPLE_DIR}/F_O_1"):
            if i.startswith("."):
                if_present_rm(f"{self.YTOS_EXAMPLE_DIR}/F_O_1/{i}")

        for i in os.listdir(f"{self.Sb2Se3_DATA_DIR}/defect"):
            if i.startswith(("O_a_", "O_b_")):
                if_present_rm(f"{self.Sb2Se3_DATA_DIR}/defect/{i}")

        if_present_rm("./vasprun.xml")

        for dir in ["bulk", "v_Cu_0", "Si_i_-1"]:
            if os.path.exists(f"{self.Cu2SiSe3_EXAMPLE_DIR}/{dir}/vasp_std/hidden_vr.gz"):
                shutil.move(
                    f"{self.Cu2SiSe3_EXAMPLE_DIR}/{dir}/vasp_std/hidden_vr.gz",
                    f"{self.Cu2SiSe3_EXAMPLE_DIR}/{dir}/vasp_std/vasprun.xml.gz",
                )

        if_present_rm(os.path.join(self.Cu2SiSe3_EXAMPLE_DIR, "Cu2SiSe3_defect_dict.json.gz"))
        if_present_rm(os.path.join(self.ZnS_DATA_DIR, "ZnS_defect_dict.json.gz"))

        if os.path.exists(f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/hidden_otcr.gz"):
            shutil.move(
                f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/hidden_otcr.gz",
                f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/OUTCAR.gz",
            )
        if_present_rm(f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/LOCPOT.gz")  # fake LOCPOT from v_Cd_-2

        if os.path.exists(f"{self.YTOS_EXAMPLE_DIR}/F_O_1/hidden_otcr.gz"):
            shutil.move(
                f"{self.YTOS_EXAMPLE_DIR}/F_O_1/hidden_otcr.gz",
                f"{self.YTOS_EXAMPLE_DIR}/F_O_1/OUTCAR.gz",
            )

        if_present_rm(f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl/another_LOCPOT.gz")
        if_present_rm(f"{self.CdTe_BULK_DATA_DIR}/another_LOCPOT.gz")
        if_present_rm(f"{self.CdTe_BULK_DATA_DIR}/another_OUTCAR.gz")
        if_present_rm(f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl/another_vasprun.xml.gz")
        if_present_rm(f"{self.CdTe_BULK_DATA_DIR}/another_vasprun.xml.gz")

        if os.path.exists(f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl/hidden_lcpt.gz"):
            shutil.move(
                f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl/hidden_lcpt.gz",
                f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl/LOCPOT.gz",
            )

    def _check_DefectsParser(self, dp, skip_corrections=False):
        # check generating thermo and plot:
        thermo = dp.get_defect_thermodynamics()
        with warnings.catch_warnings(record=True) as w:
            thermo.plot()
        assert any("You have not specified chemical potentials" in str(warn.message) for warn in w)

        # test attributes:
        assert isinstance(dp.processes, int)
        assert isinstance(dp.output_path, str)
        assert dp.skip_corrections == skip_corrections
        assert len(dp.defect_folders) >= len(dp.defect_dict)

        for name, defect_entry in dp.defect_dict.items():
            print(f"Checking {name}")
            assert name == defect_entry.name
            if defect_entry.charge_state != 0 and not skip_corrections:
                assert sum(defect_entry.corrections.values()) != 0
            assert defect_entry.get_ediff()  # can get ediff fine
            assert defect_entry.calculation_metadata  # has metadata

            # spin degeneracy is simple for our normal test cases: (others tested separately)
            if not any(x in defect_entry.name for x in ["Bipolaron", "v_Ca_+1", "v_Ca_0"]):
                assert defect_entry.degeneracy_factors[
                    "spin degeneracy"
                ] == _simple_spin_degeneracy_from_num_electrons(
                    _num_electrons_from_charge_state(
                        defect_entry.defect_supercell, defect_entry.charge_state
                    )
                )

            print(
                "Should be the same:",
                len(defect_entry.defect.equivalent_sites),
                defect_entry.defect.multiplicity,
                defect_entry.defect.get_multiplicity(symprec=dp.kwargs.get("bulk_symprec", 0.01)),
            )  # debugging
            assert len(defect_entry.defect.equivalent_sites) == defect_entry.defect.multiplicity
            assert defect_entry.defect.multiplicity == defect_entry.defect.get_multiplicity(
                symprec=dp.kwargs.get("bulk_symprec", 0.01)
            )
            assert defect_entry.defect.site in defect_entry.defect.equivalent_sites

            from pymatgen.analysis.defects.core import Substitution as pmg_Substitution
            from pymatgen.analysis.defects.core import Vacancy as pmg_Vacancy

            defect_type_dict = {
                DefectType.Vacancy: pmg_Vacancy,
                DefectType.Substitution: pmg_Substitution,
            }
            # test that custom doped multiplicity function matches pymatgen function (which is only
            # defined for Vacancies/Substitutions, and fails with periodicity-breaking cells (but
            # don't have them here with defects now defined in primitive cells, but periodicity breaking
            # supercells tested in test_generation.py)
            if defect_entry.defect.defect_type in defect_type_dict:
                assert (
                    defect_type_dict[defect_entry.defect.defect_type].get_multiplicity(defect_entry.defect)
                    == defect_entry.defect.get_multiplicity()
                )

        # check __repr__ info:
        assert all(
            i in dp.__repr__()
            for i in [
                "doped DefectsParser for bulk composition",
                f"with {len(dp.defect_dict)} parsed defect entries in self.defect_dict. "
                "Available attributes",
                "bulk_path",
                "error_tolerance",
                "Available methods",
                "get_defect_thermodynamics",
            ]
        )

    def _check_parsed_CdTe_defect_energies(self, dp):
        """
        Explicitly check some formation energies for CdTe defects.
        """
        assert np.isclose(
            dp.defect_dict["v_Cd_0"].get_ediff() - sum(dp.defect_dict["v_Cd_0"].corrections.values()),
            4.166,
            atol=3e-3,
        )  # uncorrected energy
        assert np.isclose(dp.defect_dict["v_Cd_0"].get_ediff(), 4.166, atol=1e-3)
        assert np.isclose(dp.defect_dict["v_Cd_-1"].get_ediff(), 6.355, atol=1e-3)
        assert np.isclose(
            dp.defect_dict["v_Cd_-2"].get_ediff() - sum(dp.defect_dict["v_Cd_-2"].corrections.values()),
            7.661,
            atol=3e-3,
        )  # uncorrected energy
        # slightly higher atol here, due to LOCPOT sub-sampling for file compression on repo:
        assert np.isclose(dp.defect_dict["v_Cd_-2"].get_ediff(), 8.398, atol=2e-3)
        assert np.isclose(dp.defect_dict["Int_Te_3_2"].get_ediff(), -6.2009, atol=1e-3)

    def _check_default_CdTe_DefectsParser_outputs(
        self, CdTe_dp, recorded_warnings, multiple_outcars_warning=True, dist_tol=1.5, test_attributes=True
    ):
        assert all("KPOINTS" not in str(warn.message) for warn in recorded_warnings)
        assert any(
            all(
                i in str(warn.message)
                for i in [
                    "There are mismatching INCAR tags for (some of)",
                    "in the format: 'Defects: (INCAR tag, value in defect calculation, value in bulk",
                    "['Int_Te_3_Unperturbed_1']:\n[('ADDGRID', False, True)]",
                    "In general, the same INCAR settings should be used",
                ]
            )
            for warn in recorded_warnings
        )  # INCAR warning
        if multiple_outcars_warning:
            assert any(
                all(
                    i in str(warn.message)
                    for i in [
                        "Multiple `OUTCAR` files",
                        "(directory: chosen file for parsing):",
                        f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl: OUTCAR.gz",
                        "OUTCAR files are used to",
                        "parse core levels and compute the Kumagai (eFNV) image charge correction.",
                    ]
                )
                for warn in recorded_warnings
            )
        assert any(
            "Beware: The Freysoldt (FNV) charge correction scheme has been used for some "
            "defects, while the Kumagai (eFNV) scheme has been used for others." in str(warn.message)
            for warn in recorded_warnings
        )  # multiple corrections warning
        assert all(
            any(i in str(warn.message) for warn in recorded_warnings)
            for i in [
                "Warning(s) encountered when parsing Te_Cd_+1 at ",
                "The total energies of the provided (bulk) `OUTCAR` (-218.565 eV), used to obtain the "
                "atomic core potentials for the eFNV correction, and the `vasprun.xml` (-218.518 eV), "
                "used for energies and structures, do not match. Please make sure the "
                "correct file combination is being used!",
            ]
        )  # mismatched OUTCAR and vasprun energies warning

        CdTe_thermo = CdTe_dp.get_defect_thermodynamics(dist_tol=dist_tol)
        with warnings.catch_warnings(record=True) as w:
            CdTe_thermo.plot()
        print([warn.message for warn in w])  # for debugging
        print([defect_entry.name for defect_entry in CdTe_dp.defect_dict.values()])  # for debugging
        assert any("You have not specified chemical potentials" in str(warn.message) for warn in w)
        assert any(
            "All formation energies for Int_Te_3" in str(warn.message) for warn in w
        )  # renamed to Int_Te_3_a with lowered dist_tol
        if dist_tol < 0.2:
            assert any(
                "All formation energies for Int_Te_3_Unperturbed" in str(warn.message) for warn in w
            )
        else:
            assert all(  # Int_Te_3_Unperturbed merged with Int_Te_3 with default dist_tol = 1.5
                "All formation energies for Int_Te_3_Unperturbed" not in str(warn.message) for warn in w
            )

        # test attributes:
        if test_attributes:
            assert CdTe_dp.output_path == self.CdTe_EXAMPLE_DIR
            assert CdTe_dp.dielectric == 9.13
            assert CdTe_dp.error_tolerance == 0.05
            assert CdTe_dp.bulk_path == self.CdTe_BULK_DATA_DIR  # automatically determined
            assert CdTe_dp.subfolder == "vasp_ncl"  # automatically determined
            assert CdTe_dp.bulk_band_gap_vr is None

        self._check_DefectsParser(CdTe_dp)
        assert (
            os.path.exists(os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_defect_dict.json.gz"))
            or os.path.exists(os.path.join(self.CdTe_EXAMPLE_DIR, "test_pop.json"))  # custom json name
            or os.path.exists(
                os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_test_defect_dict.json")  # custom json name
            )
        )

        self._check_parsed_CdTe_defect_energies(CdTe_dp)

        assert len(CdTe_dp.defect_folders) == 7
        for name in CdTe_dp.defect_dict:
            assert name in CdTe_dp.defect_folders  # all folder names recognised for CdTe examples

        # both OUTCARs and LOCPOTs in CdTe folders
        assert len(CdTe_dp.bulk_corrections_data) == 2
        for _k, v in CdTe_dp.bulk_corrections_data.items():
            assert v is not None

        # spot check some entries:
        assert CdTe_dp.defect_dict["v_Cd_0"].calculation_metadata["defect_site_index"] is None
        assert np.allclose(
            CdTe_dp.defect_dict["v_Cd_0"].calculation_metadata["guessed_initial_defect_site"].frac_coords,
            [0.5, 0.5, 0.5],
        )
        assert CdTe_dp.defect_dict["v_Cd_0"].calculation_metadata["bulk_site_index"] == 7
        assert CdTe_dp.defect_dict["v_Cd_-2"].calculation_metadata["guessed_defect_displacement"] is None
        assert np.allclose(
            CdTe_dp.defect_dict["v_Cd_-2"].calculation_metadata["guessed_initial_defect_site"].frac_coords,
            [0, 0, 0],
        )
        assert CdTe_dp.defect_dict["v_Cd_-2"].calculation_metadata["bulk_site_index"] == 0
        assert CdTe_dp.defect_dict["Int_Te_3_1"].calculation_metadata["defect_site_index"] == 64
        assert np.isclose(
            CdTe_dp.defect_dict["Int_Te_3_1"].calculation_metadata["guessed_defect_displacement"],
            1.45,
            atol=1e-2,
        )
        assert np.allclose(
            CdTe_dp.defect_dict["Int_Te_3_1"]
            .calculation_metadata["guessed_initial_defect_site"]
            .frac_coords,
            [0.75, 0.25, 0.75],
        )
        assert CdTe_dp.defect_dict["Int_Te_3_1"].calculation_metadata["bulk_site_index"] is None
        assert np.isclose(
            CdTe_dp.defect_dict["Int_Te_3_2"].calculation_metadata["guessed_defect_displacement"],
            1.36,
            atol=1e-2,
        )
        assert np.allclose(
            CdTe_dp.defect_dict["Int_Te_3_2"]
            .calculation_metadata["guessed_initial_defect_site"]
            .frac_coords,
            [0.9375, 0.9375, 0.6875],
        )
        assert CdTe_dp.defect_dict["Int_Te_3_2"].calculation_metadata["bulk_site_index"] is None
        assert np.isclose(
            CdTe_dp.defect_dict["Int_Te_3_Unperturbed_1"].calculation_metadata[
                "guessed_defect_displacement"
            ],
            0.93,
            atol=1e-2,
        )
        assert np.allclose(
            CdTe_dp.defect_dict["Int_Te_3_Unperturbed_1"]
            .calculation_metadata["guessed_initial_defect_site"]
            .frac_coords,
            [0.6875, 0.3125, 0.8125],
        )
        assert (
            CdTe_dp.defect_dict["Int_Te_3_Unperturbed_1"].calculation_metadata["bulk_site_index"] is None
        )
        assert CdTe_dp.defect_dict["Te_Cd_+1"].calculation_metadata["defect_site_index"] == 31
        assert np.isclose(
            CdTe_dp.defect_dict["Te_Cd_+1"].calculation_metadata["guessed_defect_displacement"],
            0.56,
            atol=1e-2,
        )
        assert np.allclose(
            CdTe_dp.defect_dict["Te_Cd_+1"]
            .calculation_metadata["guessed_initial_defect_site"]
            .frac_coords,
            [0.5, 0.5, 0.5],
        )
        assert CdTe_dp.defect_dict["Te_Cd_+1"].calculation_metadata["bulk_site_index"] == 7

        for defect_entry in CdTe_dp.defect_dict.values():
            print(defect_entry.name, defect_entry.defect.multiplicity)
            Te_i_relaxed_site_unrelaxed_struct_multiplicities = {
                "Int_Te_3_2": 24,  # C1 unrelaxed site
                "Int_Te_3_1": 12,  # Cs unrelaxed site
                "Int_Te_3_Unperturbed_1": 24,  # C1 unrelaxed site
            }
            if not any(i in defect_entry.name for i in Te_i_relaxed_site_unrelaxed_struct_multiplicities):
                assert defect_entry.defect.multiplicity == 1
            else:
                assert (
                    defect_entry.defect.multiplicity
                    == Te_i_relaxed_site_unrelaxed_struct_multiplicities[defect_entry.name]
                )

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_DefectsParser_CdTe(self):
        default_dp, w = _create_dp_and_capture_warnings(
            output_path=self.CdTe_EXAMPLE_DIR,
            dielectric=9.13,
            json_filename="CdTe_test_defect_dict.json",
        )
        self._check_default_CdTe_DefectsParser_outputs(default_dp, w)

        # test reloading DefectsParser
        reloaded_defect_dict = loadfn(os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_test_defect_dict.json"))

        for defect_name, defect_entry in reloaded_defect_dict.items():
            assert defect_entry.name == default_dp.defect_dict[defect_name].name
            assert np.isclose(defect_entry.get_ediff(), default_dp.defect_dict[defect_name].get_ediff())
            assert np.allclose(
                defect_entry.sc_defect_frac_coords,
                default_dp.defect_dict[defect_name].sc_defect_frac_coords,
            )

        # integration test using parsed CdTe thermo and chempots for plotting:
        default_thermo = default_dp.get_defect_thermodynamics(
            bulk_dos=os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_prim_k181818_NKRED_2_vasprun.xml.gz"),
        )  # test providing bulk DOS
        assert isinstance(default_thermo.bulk_dos, FermiDos)
        assert np.isclose(default_thermo.bulk_dos.get_cbm_vbm()[1], 1.65, atol=1e-2)

        return default_thermo.plot(chempots=self.CdTe_chempots, limit="CdTe-Te")

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_DefectsParser_CdTe_without_multiprocessing(self):
        # test same behaviour without multiprocessing:
        dp, w = _create_dp_and_capture_warnings(
            output_path=self.CdTe_EXAMPLE_DIR,
            dielectric=9.13,
            processes=1,
            parse_projected_eigen=False,  # just for fast testing, not recommended in general!
        )
        self._check_default_CdTe_DefectsParser_outputs(dp, w)

        # integration test using parsed CdTe thermo and chempots for plotting:
        default_thermo = dp.get_defect_thermodynamics(chempots=self.CdTe_chempots)
        return default_thermo.plot(limit="CdTe-Te")

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_DefectsParser_CdTe_filterwarnings(self):
        # check using filterwarnings works as expected:
        warnings.filterwarnings("ignore", "Multiple")
        dp, w = _create_dp_and_capture_warnings(
            output_path=self.CdTe_EXAMPLE_DIR,
            dielectric=9.13,
        )
        self._check_default_CdTe_DefectsParser_outputs(dp, w, multiple_outcars_warning=False)
        warnings.filterwarnings("default", "Multiple")

        # integration test using parsed CdTe thermo and chempots for plotting:
        default_thermo = dp.get_defect_thermodynamics(chempots=self.CdTe_chempots)
        return default_thermo.plot(limit="CdTe-Te")

    def test_DefectsParser_CdTe_dist_tol(self):
        # test with reduced dist_tol:
        # Int_Te_3_Unperturbed merged with Int_Te_3 with default dist_tol = 1.5, now no longer merged
        dp, w = _create_dp_and_capture_warnings(
            output_path=self.CdTe_EXAMPLE_DIR, dielectric=9.13, parse_projected_eigen=False
        )
        self._check_default_CdTe_DefectsParser_outputs(dp, w, dist_tol=0.1)

    @custom_mpl_image_compare(filename="CdTe_Te_Cd_+1_eigenvalue_plot.png")
    def test_DefectsParser_CdTe_no_dielectric_json(self):
        # test no dielectric and no JSON:
        dp, w = _create_dp_and_capture_warnings(
            output_path=self.CdTe_EXAMPLE_DIR,
            json_filename=False,
        )
        assert any(
            "The dielectric constant (`dielectric`) is needed to compute finite-size charge "
            "corrections, but none was provided" in str(warn.message)
            for warn in w
        )
        self._check_DefectsParser(dp, skip_corrections=True)
        assert not os.path.exists(os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_defect_dict.json.gz"))

        bes, fig = dp.defect_dict["Te_Cd_+1"].get_eigenvalue_analysis(plot=True)
        assert bes.has_unoccupied_localized_state  # has in-gap hole polaron state
        assert not any(
            [bes.has_acceptor_phs, bes.has_donor_phs, bes.has_occupied_localized_state, bes.is_shallow]
        )

        return fig

    def test_DefectsParser_CdTe_custom_settings(self):
        # test custom settings:
        dp, w = _create_dp_and_capture_warnings(
            output_path=self.CdTe_EXAMPLE_DIR,
            dielectric=[9.13, 9.13, 9.13],
            error_tolerance=0.01,
            skip_corrections=False,
            bulk_band_gap_vr=f"{self.CdTe_BULK_DATA_DIR}/vasprun.xml",
            processes=4,
            json_filename="test_pop.json",
        )
        assert any(
            all(
                i in str(warn.message)
                for i in [
                    "Estimated error in the Kumagai (eFNV) charge correction for certain defects",
                    "greater than the `error_tolerance` (= 0.010 eV):",
                    "Int_Te_3_2: 0.012 eV",
                    "You may want to check the accuracy",
                ]
            )
            for warn in w
        )  # correction warning
        assert os.path.exists(os.path.join(self.CdTe_EXAMPLE_DIR, "test_pop.json"))
        self._check_default_CdTe_DefectsParser_outputs(dp, w, test_attributes=False)  # same energies as
        # above

        # test changed attributes:
        assert dp.output_path == self.CdTe_EXAMPLE_DIR
        assert dp.dielectric == [9.13, 9.13, 9.13]
        assert dp.error_tolerance == 0.01
        assert isinstance(dp.bulk_band_gap_vr, Vasprun)
        assert dp.processes == 4
        assert dp.json_filename == "test_pop.json"

    def test_DefectsParser_CdTe_unrecognised_subfolder(self):
        # test setting subfolder to unrecognised one:
        with pytest.raises(FileNotFoundError) as exc:
            DefectsParser(
                output_path=self.CdTe_EXAMPLE_DIR,
                subfolder="vasp_std",
            )
        assert (
            f"`vasprun.xml(.gz)` files (needed for defect parsing) not found in bulk folder at: "
            f"{self.CdTe_EXAMPLE_DIR}/CdTe_bulk or subfolder: vasp_std -- please ensure "
            f"`vasprun.xml(.gz)` files are present and/or specify `bulk_path` manually."
        ) in str(exc.value)

    def test_DefectsParser_CdTe_skip_corrections(self):
        # skip_corrections:
        dp, _w = _create_dp_and_capture_warnings(
            output_path=self.CdTe_EXAMPLE_DIR, skip_corrections=True, parse_projected_eigen=False
        )
        self._check_DefectsParser(dp, skip_corrections=True)

    def test_DefectsParser_CdTe_aniso_dielectric(self):
        # anisotropic dielectric
        fake_aniso_dielectric = [1, 2, 3]
        dp, w = _create_dp_and_capture_warnings(
            output_path=self.CdTe_EXAMPLE_DIR,
            bulk_path="CdTe_bulk",
            dielectric=fake_aniso_dielectric,
        )
        assert not any(  # no correction warning now for Int_Te_3_2, as it's unstable here
            all(
                i in str(warn.message)
                for i in [
                    "Estimated error in the Kumagai (eFNV) charge correction for certain defects",
                    "greater than the `error_tolerance` (= 0.050 eV):",
                    "Int_Te_3_2: 0.157 eV",
                    "You may want to check the accuracy",
                ]
            )
            for warn in w
        )  # correction warning

        assert any(
            f"Defects: {i} each encountered the same warning:" in str(warn.message)
            for warn in w
            for i in ["{'v_Cd_-2', 'v_Cd_-1'}", "{'v_Cd_-1', 'v_Cd_-2'}"]
        )

        for i in [
            "An anisotropic dielectric constant was supplied, but `OUTCAR` files (needed to compute the "
            "_anisotropic_ Kumagai eFNV charge correction) are missing from the defect or bulk folder.",
            "`LOCPOT` files were found in both defect & bulk folders, and so the Freysoldt (FNV) "
            "charge correction developed for _isotropic_ materials will be applied here, "
            "which corresponds to using the effective isotropic average of the supplied "
            "anisotropic dielectric. This could lead to significant errors for very anisotropic "
            "systems and/or relatively small supercells!",
            f"(using bulk path {self.CdTe_EXAMPLE_DIR}/CdTe_bulk/vasp_ncl and vasp_ncl defect "
            f"subfolders)",
        ]:
            assert any(i in str(warn.message) for warn in w)

        assert any(
            all(
                i in str(warn.message)
                for i in [
                    "An anisotropic dielectric constant was supplied, but `OUTCAR` files (needed to "
                    "compute the _anisotropic_ Kumagai eFNV charge correction) are missing from the "
                    "defect or bulk folder.",
                    "`LOCPOT` files were found in both defect & bulk folders, and so the Freysoldt (FNV) "
                    "charge correction developed for _isotropic_ materials will be applied here, "
                    "which corresponds to using the effective isotropic average of the supplied "
                    "anisotropic dielectric. This could lead to significant errors for very anisotropic "
                    "systems and/or relatively small supercells!",
                    f"(using bulk path {self.CdTe_EXAMPLE_DIR}/CdTe_bulk/vasp_ncl and vasp_ncl defect "
                    f"subfolders)",
                ]
            )
            for warn in w
        )
        self._check_DefectsParser(dp)

    def test_DefectsParser_CdTe_kpoints_mismatch(self):
        dp, w = _create_dp_and_capture_warnings(
            output_path=self.CdTe_EXAMPLE_DIR,
            bulk_path=f"{self.module_path}/data/CdTe",  # vasp_gam bulk vr here
            dielectric=9.13,
            parse_projected_eigen=False,  # just for fast testing, not recommended in general!
        )

        for i in [
            "Defects: ",  # multiple warnings here so best to do this way:
            "'Int_Te_3_1'",
            "'v_Cd_-2'",
            "'v_Cd_-1'",
            "'Te_Cd_+1'",
            "'Int_Te_3_2'",
            "'Int_Te_3_Unperturbed_1'",
            "each encountered the same warning:",
            "`LOCPOT` or `OUTCAR` files are missing from the defect or bulk folder. These are needed to",
            "and vasp_ncl defect subfolders)",
            "There are mismatching INCAR tags for (some of) your defect and bulk calculations",
            "There are mismatching KPOINTS for (some of) your defect and bulk calculations ",
            "Found the following differences:",
            "(in the format: (defect kpoints, bulk kpoints)):",
            "Int_Te_3_1: [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, "
            "0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]], [[0.0, 0.0, 0.0]]]",
            "v_Cd_0: [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.5],",
            "Int_Te_3_Unperturbed_1: [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.5],",
            "In general, the same KPOINTS settings should be used",
        ]:
            assert any(i in str(warn.message) for warn in w)
        dp.get_defect_thermodynamics()  # test thermo generation works fine

    def test_DefectsParser_corrections_errors_warning(self):
        _dp, w = _create_dp_and_capture_warnings(
            output_path=self.CdTe_EXAMPLE_DIR,
            dielectric=9.13,
            error_tolerance=0.001,
        )  # low error tolerance to force warnings

        for i in [
            "Estimated error in the Freysoldt (FNV) ",
            "Estimated error in the Kumagai (eFNV) ",
            "charge correction for certain defects is greater than the `error_tolerance` (= 1.00e-03 eV):",
            "v_Cd_-2: 1.08e-02 eV",
            # "v_Cd_-1: 8.46e-03 eV",  # now not printed because not stable charge states
            "Int_Te_3_1: 3.10e-03 eV",
            "Te_Cd_+1: 2.02e-03 eV",
            # "Int_Te_3_Unperturbed_1: 4.91e-03 eV",  # now not printed because not stable charge states
            "Int_Te_3_2: 1.24e-02 eV",
            "You may want to check the accuracy of the corrections by",
            "(using `defect_entry.get_freysoldt_correction()` with `plot=True`)",
            "(using `defect_entry.get_kumagai_correction()` with `plot=True`)",
        ]:
            assert any(i in str(warn.message) for warn in w)  # correction errors warnings

    @custom_mpl_image_compare(filename="YTOS_example_defects_plot.png")
    def test_DefectsParser_YTOS_default_bulk(self):
        dp, w = _create_dp_and_capture_warnings(
            output_path=self.YTOS_EXAMPLE_DIR,
            dielectric=self.ytos_dielectric,
            json_filename="YTOS_example_defect_dict.json",
        )  # for testing in test_thermodynamics.py
        assert not w
        self._check_DefectsParser(dp)

        # spot check some multiplicities:
        assert dp.defect_dict["F_O_1"].defect.multiplicity == 1  # O_D4h site
        assert np.allclose(dp.defect_dict["F_O_1"].defect.site.frac_coords, [0, 0, 0])
        assert dp.defect_dict["Int_F_-1"].defect.multiplicity == 2  # old C4v site (see comments below)

        thermo = dp.get_defect_thermodynamics()
        dumpfn(
            thermo, os.path.join(self.YTOS_EXAMPLE_DIR, "YTOS_example_thermo.json")
        )  # for test_plotting
        return thermo.plot()  # no chempots for YTOS formation energy plot test

    @custom_mpl_image_compare(filename="YTOS_example_defects_plot.png")
    def test_DefectsParser_YTOS_macOS_duplicated_OUTCAR(self):
        with open(f"{self.YTOS_EXAMPLE_DIR}/F_O_1/._OUTCAR", "w") as f:
            f.write("test pop")
        with open(f"{self.YTOS_EXAMPLE_DIR}/F_O_1/._vasprun.xml", "w") as f:
            f.write("test pop")
        with open(f"{self.YTOS_EXAMPLE_DIR}/F_O_1/._LOCPOT", "w") as f:
            f.write("test pop")
        with open(f"{self.YTOS_EXAMPLE_DIR}/F_O_1/.DS_Store", "w") as f:
            f.write("test pop")

        dp, w = _create_dp_and_capture_warnings(
            output_path=self.YTOS_EXAMPLE_DIR,
            dielectric=self.ytos_dielectric,
            json_filename="YTOS_example_defect_dict.json",
        )  # for testing in test_thermodynamics.py
        assert not w  # hidden files ignored
        self._check_DefectsParser(dp)
        thermo = dp.get_defect_thermodynamics()
        return thermo.plot()  # no chempots for YTOS formation energy plot test

    @custom_mpl_image_compare(filename="YTOS_example_defects_plot.png")
    def test_DefectsParser_YTOS_macOS_duplicated_bulk_OUTCAR(self):
        with open(f"{self.YTOS_EXAMPLE_DIR}/Bulk/._OUTCAR", "w") as f:
            f.write("test pop")
        with open(f"{self.YTOS_EXAMPLE_DIR}/Bulk/._vasprun.xml", "w") as f:
            f.write("test pop")
        with open(f"{self.YTOS_EXAMPLE_DIR}/Bulk/._LOCPOT", "w") as f:
            f.write("test pop")
        with open(f"{self.YTOS_EXAMPLE_DIR}/F_O_1/.DS_Store", "w") as f:
            f.write("test pop")
        dp, w = _create_dp_and_capture_warnings(
            output_path=self.YTOS_EXAMPLE_DIR,
            dielectric=self.ytos_dielectric,
            json_filename="YTOS_example_defect_dict.json",
            parse_projected_eigen=False,  # just for fast testing, not recommended in general!
        )  # for testing in test_thermodynamics.py
        assert not w  # hidden files ignored
        self._check_DefectsParser(dp)
        thermo = dp.get_defect_thermodynamics()
        return thermo.plot()  # no chempots for YTOS formation energy plot test

    @custom_mpl_image_compare(filename="YTOS_example_defects_plot.png")
    def test_DefectsParser_YTOS_explicit_bulk(self):
        dp, w = _create_dp_and_capture_warnings(
            output_path=self.YTOS_EXAMPLE_DIR,
            bulk_path=os.path.join(self.YTOS_EXAMPLE_DIR, "Bulk"),
            dielectric=self.ytos_dielectric,
            parse_projected_eigen=False,  # just for fast testing, not recommended in general!
        )
        assert not w
        self._check_DefectsParser(dp)
        thermo = dp.get_defect_thermodynamics()
        return thermo.plot()  # no chempots for YTOS formation energy plot test

    def test_DefectsParser_no_defects_parsed_error(self):
        with pytest.raises(ValueError) as exc:
            DefectsParser(
                output_path=self.YTOS_EXAMPLE_DIR,
                subfolder="vasp_gam",
            )
        assert (
            f"No defect calculations in `output_path` '{self.YTOS_EXAMPLE_DIR}' were successfully parsed, "
            f"using `bulk_path`: {self.YTOS_EXAMPLE_DIR}/Bulk and `subfolder`: 'vasp_gam'. Please check "
            f"the correct defect/bulk paths and subfolder are being set, and that the folder structure is "
            f"as expected (see `DefectsParser` docstring)." in str(exc.value)
        )

    @custom_mpl_image_compare(filename="O_Se_example_defects_plot.png")
    def test_extrinsic_Sb2Se3(self):
        with pytest.raises(ValueError) as exc:
            DefectsParser(
                output_path=f"{self.Sb2Se3_DATA_DIR}/defect",
                dielectric=self.Sb2Se3_dielectric,
            )
        assert (  # bulk in separate folder so fails
            f"Could not automatically determine bulk supercell calculation folder in "
            f"{self.Sb2Se3_DATA_DIR}/defect, found 0 folders containing `vasprun.xml(.gz)` files (in "
            f"subfolders) and 'bulk' in the folder name" in str(exc.value)
        )

        # no warning about negative corrections with strong anisotropic dielectric:
        Sb2Se3_O_dp, w = _create_dp_and_capture_warnings(
            output_path=f"{self.Sb2Se3_DATA_DIR}/defect",
            bulk_path=f"{self.Sb2Se3_DATA_DIR}/bulk",
            dielectric=self.Sb2Se3_dielectric,
            json_filename="Sb2Se3_O_example_defect_dict.json",
        )  # for testing in test_thermodynamics.py
        assert not w  # no warnings
        self._check_DefectsParser(Sb2Se3_O_dp)
        Sb2Se3_O_thermo = Sb2Se3_O_dp.get_defect_thermodynamics()
        dumpfn(
            Sb2Se3_O_thermo, os.path.join(self.Sb2Se3_DATA_DIR, "Sb2Se3_O_example_thermo.json")
        )  # for test_plotting

        # warning about negative corrections when using (fake) isotropic dielectric:
        Sb2Se3_O_dp, w = _create_dp_and_capture_warnings(
            output_path=f"{self.Sb2Se3_DATA_DIR}/defect",
            bulk_path=f"{self.Sb2Se3_DATA_DIR}/bulk",
            dielectric=40,  # fake isotropic dielectric
            parse_projected_eigen=False,  # just for fast testing, not recommended in general!
        )
        assert any(
            all(
                i in str(warn.message)
                for i in [
                    "The calculated finite-size charge corrections for defect at",
                    "sum to a _negative_ value of -0.144.",
                ]
            )
            for warn in w
        )

        # spot check:
        assert np.isclose(Sb2Se3_O_thermo.get_formation_energy("O_Se_Cs_Sb2.65_-2"), -1.84684, atol=1e-3)

        return Sb2Se3_O_thermo.plot(chempots={"O": -8.9052, "Se": -5})  # example chempots

    def test_extrinsic_Sb2Se3_parsing_with_single_defect_dir(self):
        # no warning about negative corrections with strong anisotropic dielectric:
        Sb2Se3_O_dp, w = _create_dp_and_capture_warnings(
            output_path=f"{self.Sb2Se3_DATA_DIR}/defect/O_-2",
            bulk_path=f"{self.Sb2Se3_DATA_DIR}/bulk",
            dielectric=self.Sb2Se3_dielectric,
        )
        assert not w  # no warnings
        self._check_DefectsParser(Sb2Se3_O_dp)
        Sb2Se3_O_thermo = Sb2Se3_O_dp.get_defect_thermodynamics()
        assert np.isclose(Sb2Se3_O_thermo.get_formation_energy("O_Se_Cs_Sb2.65_-2"), -1.84684, atol=1e-3)

        assert len(Sb2Se3_O_thermo.defect_entries) == 1  # only the one specified defect parsed

    def test_duplicate_folders_Sb2Se3(self):
        shutil.copytree(f"{self.Sb2Se3_DATA_DIR}/defect/O_2", f"{self.Sb2Se3_DATA_DIR}/defect/O_a_2")
        shutil.copytree(f"{self.Sb2Se3_DATA_DIR}/defect/O_2", f"{self.Sb2Se3_DATA_DIR}/defect/O_b_2")
        shutil.copytree(f"{self.Sb2Se3_DATA_DIR}/defect/O_2", f"{self.Sb2Se3_DATA_DIR}/defect/O_a_1")
        shutil.copytree(f"{self.Sb2Se3_DATA_DIR}/defect/O_1", f"{self.Sb2Se3_DATA_DIR}/defect/O_b_1")
        Sb2Se3_O_dp, w = _create_dp_and_capture_warnings(
            output_path=f"{self.Sb2Se3_DATA_DIR}/defect",
            bulk_path=f"{self.Sb2Se3_DATA_DIR}/bulk",
            dielectric=self.Sb2Se3_dielectric,
        )  # for testing in test_thermodynamics.py
        assert any(
            "The following parsed defect entries were found to be duplicates (exact same defect "
            "supercell energies)" in str(warn.message)
            for warn in w
        )
        assert any(
            "[O_Se_Cs_Sb2.65_+2 (O_2), O_Se_Cs_Sb2.65_+2 (O_a_1), O_Se_Cs_Sb2.65_+2 (O_a_2), "
            "O_Se_Cs_Sb2.65_+2 (O_b_2)]\n[O_Se_Cs_Sb2.65_+1 (O_1), O_Se_Cs_Sb2.65_+1 (O_b_1)]"
            in str(warn.message)
            for warn in w
        )
        self._check_DefectsParser(Sb2Se3_O_dp)

    @custom_mpl_image_compare(filename="Sb2Si2Te6_v_Sb_-3_eFNV_plot_no_intralayer.png")
    def test_sb2si2te6_eFNV(self):
        dp, w = _create_dp_and_capture_warnings(
            self.Sb2Si2Te6_EXAMPLE_DIR,
            dielectric=self.Sb2Si2Te6_dielectric,
            json_filename="Sb2Si2Te6_example_defect_dict.json",  # testing in test_thermodynamics.py
            parse_projected_eigen=False,  # just for fast testing, not recommended in general!
        )
        assert any(
            "Estimated error in the Kumagai (eFNV) charge correction for certain defects"
            in str(warning.message)
            for warning in w
        )  # collated warning
        assert all(
            "Estimated error in the Kumagai (eFNV) charge correction for defect"
            not in str(warning.message)
            for warning in w
        )  # no individual-level warning
        # Sb2Si2Te6 supercell breaks periodicity, but we don't throw warning when just parsing defects
        assert all("The defect supercell has been detected" not in str(warning.message) for warning in w)

        self._check_DefectsParser(dp)

        sb2si2te6_thermo = dp.get_defect_thermodynamics()
        dumpfn(sb2si2te6_thermo, os.path.join(self.Sb2Si2Te6_EXAMPLE_DIR, "Sb2Si2Te6_example_thermo.json"))
        with warnings.catch_warnings(record=True) as w:
            sb2si2te6_thermo.get_symmetries_and_degeneracies()
        print([str(warning.message) for warning in w])
        assert any(_orientational_degeneracy_warning in str(warning.message) for warning in w)

        v_Sb_minus_3_ent = dp.defect_dict["v_Sb_-3"]
        with warnings.catch_warnings(record=True) as w:
            correction, fig = v_Sb_minus_3_ent.get_kumagai_correction(plot=True)
        assert any(
            "Estimated error in the Kumagai (eFNV) charge correction for defect v_Sb_-3 is 0.067 eV (i.e. "
            "which is greater than the `error_tolerance`: 0.050 eV)." in str(warn.message)
            for warn in w
        )
        assert np.isclose(correction.correction_energy, 1.077, atol=1e-3)
        assert np.isclose(
            v_Sb_minus_3_ent.corrections_metadata.get("kumagai_charge_correction_error", 0),
            0.067,
            atol=1e-3,
        )

        with warnings.catch_warnings(record=True) as w:
            correction, fig = v_Sb_minus_3_ent.get_kumagai_correction(plot=True, defect_region_radius=8.75)
        assert not any("Estimated error" in str(warn.message) for warn in w)
        assert np.isclose(correction.correction_energy, 1.206, atol=1e-3)
        assert np.isclose(
            v_Sb_minus_3_ent.corrections_metadata.get("kumagai_charge_correction_error", 0),
            0.023,
            atol=1e-3,
        )

        # get indices of sites within 3 Å of the defect site when projected along the _a_ lattice vector
        # (inter-layer direction in our supercell)
        sites_within_3A = [
            i
            for i, site in enumerate(v_Sb_minus_3_ent.defect_supercell)
            if abs(site.frac_coords[0] - v_Sb_minus_3_ent.defect_supercell_site.frac_coords[0]) < 0.2
        ]
        with warnings.catch_warnings(record=True) as w:
            correction, fig = v_Sb_minus_3_ent.get_kumagai_correction(
                plot=True, excluded_indices=sites_within_3A
            )
        assert not any("Estimated error" in str(warn.message) for warn in w)
        assert np.isclose(correction.correction_energy, 1.234, atol=1e-3)
        assert np.isclose(
            v_Sb_minus_3_ent.corrections_metadata.get("kumagai_charge_correction_error", 0),
            0.017,
            atol=1e-3,
        )

        return fig

    @custom_mpl_image_compare(filename="neutral_v_O_plot.png")
    def test_V2O5_FNV(self):
        # only three inequivalent neutral V_O present
        dp, w = _create_dp_and_capture_warnings(
            self.V2O5_DATA_DIR,
            dielectric=[4.186, 19.33, 17.49],
            json_filename="V2O5_example_defect_dict.json",  # testing in test_thermodynamics.py
        )
        assert not w  # no warnings
        assert len(dp.defect_dict) == 3  # only three inequivalent neutral V_O present

        self._check_DefectsParser(dp)

        v2o5_chempots = loadfn(os.path.join(self.V2O5_DATA_DIR, "chempots.json"))
        v2o5_thermo = dp.get_defect_thermodynamics(chempots=v2o5_chempots)
        dumpfn(v2o5_thermo, os.path.join(self.V2O5_DATA_DIR, "V2O5_example_thermo.json"))

        with warnings.catch_warnings(record=True) as w:
            v2o5_thermo.get_symmetries_and_degeneracies()
        print([str(warning.message) for warning in w])
        assert not w  # no warnings

        return v2o5_thermo.plot(limit="V2O5-O2")

    @custom_mpl_image_compare(filename="merged_renamed_v_O_plot.png")
    def test_V2O5_same_named_defects(self):
        shutil.copytree(self.V2O5_DATA_DIR, "V2O5_test")
        shutil.copytree("V2O5_test/v_O_1", "V2O5_test/unrecognised_4")
        shutil.copytree("V2O5_test/v_O_1", "V2O5_test/unrecognised_5")
        for i in os.listdir("V2O5_test"):
            if os.path.isdir(f"V2O5_test/{i}") and i.startswith("v_O"):
                shutil.move(f"V2O5_test/{i}", f"V2O5_test/unrecognised_{i[-1]}")

        dp, w = _create_dp_and_capture_warnings("V2O5_test", dielectric=[4.186, 19.33, 17.49])
        assert any(
            "The following parsed defect entries were found to be duplicates" in str(warning.message)
            for warning in w
        )
        assert any(
            "v_O_Cs_V1.60_0 (unrecognised_1), v_O_Cs_V1.60_0 (unrecognised_4), v_O_Cs_V1.60_0 ("
            "unrecognised_5)" in str(warning.message)
            for warning in w
        )
        assert len(dp.defect_dict) == 3  # only 3 defects, 2 duplicates warned and omitted
        self._check_DefectsParser(dp)
        thermo = dp.get_defect_thermodynamics()
        v2o5_chempots = loadfn(os.path.join(self.V2O5_DATA_DIR, "chempots.json"))
        thermo.chempots = v2o5_chempots

        print(thermo.get_symmetries_and_degeneracies())

        return thermo.plot(limit="V2O5-O2")

    @custom_mpl_image_compare(filename="SrTiO3_v_O.png")
    def test_SrTiO3_diff_ISYM_bulk_defect_and_concentration_funcs(self):
        """
        Test parsing SrTiO3 defect calculations, where a different ISYM was
        used for the bulk (= 3) compared to the defect (= 0) calculations, as
        well as the ``DefectThermodynamics`` concentration functions with
        various options.

        Previously this failed because the bulk/defect kpoints could not be
        properly compared.
        """
        # test previous parsing approach first:
        with warnings.catch_warnings(record=True) as w:
            _single_dp = DefectParser.from_paths(
                defect_path=f"{self.SrTiO3_DATA_DIR}/vac_O_2/vasp_std",
                bulk_path=f"{self.SrTiO3_DATA_DIR}/bulk_sp333",
                dielectric=6.33,
            )
        print([str(warning.message) for warning in w])  # for debugging
        assert len(w) == 1
        assert all(
            i in str(w[0].message)
            for i in [
                "There are mismatching INCAR tags",
                "[('LASPH', True, False)]",
            ]
        )

        dp, w = _create_dp_and_capture_warnings(
            self.SrTiO3_DATA_DIR, dielectric=6.33, parse_projected_eigen=False
        )  # wrong dielectric from Kanta
        print(dp.defect_dict.keys())
        assert len(w) == 1
        assert all(
            i in str(w[0].message)
            for i in [
                "There are mismatching INCAR tags",
                "['vac_O_0', 'vac_O_1', 'vac_O_2']:\n[('LASPH', True, False)]",
            ]
        )

        assert len(dp.defect_dict) == 3
        self._check_DefectsParser(dp)

        # some hardcoded symmetry tests with default `symprec = 0.1` for relaxed structures:
        assert dp.defect_dict["vac_O_2"].calculation_metadata["relaxed point symmetry"] == "C2v"
        assert dp.defect_dict["vac_O_1"].calculation_metadata["relaxed point symmetry"] == "Cs"
        assert dp.defect_dict["vac_O_0"].calculation_metadata["relaxed point symmetry"] == "C2v"
        thermo = dp.get_defect_thermodynamics()

        # hardcoded check of bulk_site_concentration property:
        sto_O_site_conc = 5.067520709900586e22
        for defect_entry in dp.defect_dict.values():  # oxygen site concentration
            assert np.isclose(defect_entry.bulk_site_concentration, sto_O_site_conc, rtol=1e-4)

        print(thermo.get_symmetries_and_degeneracies())

        conc_df = thermo.get_equilibrium_concentrations()  # no chempots or Fermi level
        print("conc_df", conc_df)  # for debugging
        srtio3_V_O_conc_lists = [  # with no chempots or Fermi level (so using Eg/2)
            ["4.456e-141", 9.742, "N/A", "100.00%"],  # +2  # "N/A" is placeholder here for per-site concs
            ["2.497e-162", 11.043, "N/A", "0.00%"],  # +1
            ["1.109e-189", 12.635, "N/A", "0.00%"],  # 0
        ]  # (in order of positive to negative, left to right on formation energy diagram)
        for i, (index, row) in enumerate(conc_df.iterrows()):
            print(i, index, row)
            assert list(row) == [srtio3_V_O_conc_lists[i][j] for j in [0, 1, 3]]  # skip "N/A" for per-site

        for kwargs in [{"per_site": True}, {"per_site": True, "skip_formatting": True}]:
            per_site_conc_df = thermo.get_equilibrium_concentrations(**kwargs)
            print("per_site_conc_df", per_site_conc_df)  # for debugging
            for i, (index, row) in enumerate(per_site_conc_df.iterrows()):
                print(i, index, row)
                assert isinstance(index[-1], int if kwargs.get("skip_formatting") else str)  # charge type?
                for j, (col_name, val) in enumerate(row.items()):
                    if col_name != "Concentration (per site)":  # per-site concentration
                        if isinstance(val, str):
                            assert val == srtio3_V_O_conc_lists[i][j]
                        else:
                            assert np.isclose(val, float(srtio3_V_O_conc_lists[i][j]))
                    elif kwargs.get("skip_formatting"):
                        assert np.isclose(val, float(srtio3_V_O_conc_lists[i][0]) / sto_O_site_conc)
                    else:
                        assert np.isclose(
                            float(val[:-2]), 100 * float(srtio3_V_O_conc_lists[i][0]) / sto_O_site_conc
                        )

        assert thermo.get_equilibrium_concentrations(per_charge=False).to_numpy().tolist() == [
            ["4.456e-141"]
        ]
        print(  # for debugging
            "per_charge_F conc_df:", thermo.get_equilibrium_concentrations(per_charge=False, per_site=True)
        )
        for skip_formatting in [True, False]:
            per_site_conc = (
                thermo.get_equilibrium_concentrations(
                    per_charge=False, per_site=True, skip_formatting=skip_formatting
                )["Concentration (per site)"]
                .to_numpy()
                .tolist()[0]
            )
            assert np.isclose(
                per_site_conc if skip_formatting else float(per_site_conc[:-2]),
                4.456e-141 / sto_O_site_conc * (1 if skip_formatting else 100),
                rtol=1e-3,
            )

        assert next(
            iter(
                thermo.get_equilibrium_concentrations(per_charge=False, fermi_level=1.710795)
                .to_numpy()
                .tolist()
            )
        ) == ["2.004e-142"]

        per_site_conc_df = thermo.get_equilibrium_concentrations(per_site=True, fermi_level=1.710795)
        print("per_site_conc_df", per_site_conc_df)  # for debugging
        custom_fermi_concs = ["3.954e-163 %", "1.045e-183 %", "2.189e-210 %"]
        for i, (index, row) in enumerate(per_site_conc_df.iterrows()):
            print(i, index, row)
            assert row["Concentration (per site)"] == custom_fermi_concs[i]

        # test get_fermi_level_and_concentrations
        fermi_level, e_conc, h_conc, conc_df = thermo.get_fermi_level_and_concentrations(
            bulk_dos=f"{self.SrTiO3_DATA_DIR}/bulk_sp333/vasprun.xml",
            annealing_temperature=300,
            per_charge=False,
        )
        print("conc_df", conc_df)  # for debugging
        assert np.isclose(fermi_level, 1.710795, rtol=1e-4)
        assert np.isclose(e_conc, h_conc, rtol=1e-4)  # same because defect concentration negligible
        # without chempots
        assert np.isclose(e_conc, 6.129e-7, rtol=1e-3)
        assert conc_df.to_numpy().tolist() == [["2.004e-142"]]
        assert conc_df.index[0] == "vac_O"
        assert conc_df.index.name == "Defect"

        fermi_level, e_conc, h_conc, conc_df = thermo.get_fermi_level_and_concentrations(
            bulk_dos=f"{self.SrTiO3_DATA_DIR}/bulk_sp333/vasprun.xml",
            annealing_temperature=300,
            skip_formatting=True,
        )
        print("quenched conc_df", conc_df)  # for debugging
        assert np.isclose(fermi_level, 1.710795, rtol=1e-4)
        assert np.isclose(e_conc, h_conc, rtol=1e-4)  # same because defect concentration negligible
        # without chempots
        assert np.isclose(e_conc, 6.129e-7, rtol=1e-3)
        quenched_conc_df_lists = [
            [2.003657893378957e-142, 9.822, "100.00%", 2.003657893378957e-142],  # +2
            [5.294723641800535e-163, 11.083, "0.00%", 2.003657893378957e-142],  # +1
            [1.10921226234365e-189, 12.635, "0.00%", 2.003657893378957e-142],  # 0
        ]
        for i, row in enumerate(quenched_conc_df_lists):
            print(i, row)
            assert list(conc_df.iloc[i]) == row

        fermi_level, e_conc, h_conc, conc_df = thermo.get_fermi_level_and_concentrations(
            bulk_dos=f"{self.SrTiO3_DATA_DIR}/bulk_sp333/vasprun.xml",
            annealing_temperature=300,
            per_site=True,
        )
        print("per_site quenched conc_df", conc_df)  # for debugging
        assert np.isclose(fermi_level, 1.710795, rtol=1e-4)
        assert np.isclose(e_conc, h_conc, rtol=1e-4)  # same because defect concentration negligible
        # without chempots
        assert np.isclose(e_conc, 6.129e-7, rtol=1e-3)
        quenched_per_site_conc_df_lists = [
            ["2.004e-142", 9.822, "3.954e-163 %", "100.00%", "2.004e-142"],  # +2
            ["5.295e-163", 11.083, "1.045e-183 %", "0.00%", "2.004e-142"],  # +1
            ["1.109e-189", 12.635, "2.189e-210 %", "0.00%", "2.004e-142"],  # 0
        ]
        for i, row in enumerate(quenched_per_site_conc_df_lists):
            print(i, row)
            assert list(conc_df.iloc[i]) == row

        fermi_level, e_conc, h_conc, conc_df = thermo.get_fermi_level_and_concentrations(
            bulk_dos=f"{self.SrTiO3_DATA_DIR}/bulk_sp333/vasprun.xml",
            annealing_temperature=300,
            per_site=True,
            per_charge=False,
            skip_formatting=True,
        )
        print("per_site not per_charge quenched conc_df", conc_df)  # for debugging
        assert np.isclose(fermi_level, 1.710795, rtol=1e-4)
        assert np.isclose(e_conc, h_conc, rtol=1e-4)  # same because defect concentration negligible
        # without chempots
        assert np.isclose(e_conc, 6.129e-7, rtol=1e-3)
        assert conc_df.to_numpy().tolist()[0][-1] == 3.9539214688363133e-165
        assert conc_df.index.to_numpy()[0] == "vac_O"
        assert conc_df.index.name == "Defect"

        return thermo.plot()

    @custom_mpl_image_compare(filename="ZnS_defects.png")
    def test_ZnS_non_diagonal_NKRED_mismatch(self):
        """
        Test parsing ZnS defect calculations, which were performed with a non-
        diagonal periodicity-breaking supercell, and with NKRED mismatch from
        defect and bulk supercells.
        """
        dp, w = _create_dp_and_capture_warnings(self.ZnS_DATA_DIR, dielectric=8.9)
        assert len(w) == 1
        assert all(
            i in str(w[0].message)
            for i in [
                "There are mismatching INCAR tags",
                ":\n[('NKRED', 1, 2)]\nIn",
            ]
        )
        assert str(w[0].message).count(":\n[('NKRED', 1, 2)]\nIn") == 1  # only once

        assert len(dp.defect_dict) == 17
        self._check_DefectsParser(dp)
        thermo = dp.get_defect_thermodynamics()

        with warnings.catch_warnings(record=True) as w:
            symm_df = thermo.get_symmetries_and_degeneracies()
        print(symm_df)  # for debugging
        print([str(warning.message) for warning in w])  # for debugging
        assert len(w) == 1
        assert all(
            i in str(w[0].message)
            for i in [
                "The defect supercell has been detected to possibly have a non-scalar matrix expansion",
                "breaking the cell periodicity",
                "This will not affect defect formation energies / transition levels,",
                "but is important for concentrations/doping/Fermi level behaviour",
                "You can manually check (and edit) the computed defect/bulk point",
            ]
        )

        vacancy_and_sub_rows = symm_df[
            np.array(["vac" in i for i in symm_df.index.get_level_values("Defect")])
            | np.array(["sub" in i for i in symm_df.index.get_level_values("Defect")])
        ]
        assert list(vacancy_and_sub_rows["Site_Symm"].unique()) == ["Td"]
        assert list(vacancy_and_sub_rows["Defect_Symm"].unique()) == ["C1"]

        interstitial_rows = symm_df[["inter" in i for i in symm_df.index.get_level_values("Defect")]]
        assert list(interstitial_rows["Site_Symm"].unique()) == ["C3v", "Cs", "C1"]
        assert list(interstitial_rows["Defect_Symm"].unique()) == ["C1"]

        thermo.dist_tol = 2.5  # merges Al interstitials together
        # remove eigenvalue_data and run_metadata from each entry to save space:
        for defect_entry in thermo.defect_entries.values():
            defect_entry.calculation_metadata["eigenvalue_data"] = None
            defect_entry.calculation_metadata["run_metadata"] = None
        thermo.to_json(os.path.join(self.ZnS_DATA_DIR, "ZnS_thermo.json"))
        return thermo.plot()

    def test_solid_solution_oxi_state_handling(self):
        """
        Test parsing a defect in a large, complex solid solution supercell,
        which hangs with using ``pymatgen``'s oxi state methods (so is set as
        'undetermined' by ``doped``, as this property isn't necessary when
        parsing).
        """
        # no warning with no dielectric/OUTCARs, as is neutral
        dp, w = _create_dp_and_capture_warnings(self.SOLID_SOLUTION_DATA_DIR, parse_projected_eigen=False)
        assert not w
        assert len(dp.defect_dict) == 1
        self._check_DefectsParser(dp)
        thermo = dp.get_defect_thermodynamics()

        with warnings.catch_warnings(record=True) as w:
            symm_df = thermo.get_symmetries_and_degeneracies()
        print([str(warning.message) for warning in w])  # for debugging
        assert not w

        assert list(symm_df["Site_Symm"].unique()) == ["C1"]
        assert list(symm_df["Defect_Symm"].unique()) == ["C1"]
        assert list(symm_df["g_Orient"].unique()) == [1.0]
        assert list(symm_df["g_Spin"].unique()) == [2]

    @custom_mpl_image_compare(filename="CaO_v_Ca.png")
    def test_CaO_symmetry_determination(self):
        """
        Test parsing CaO defect calculations, and confirming the correct point
        group symmetries are being determined (previously failed with old
        relaxed symmetry determination scheme with old default of
        ``symprec=0.2``).
        """
        dp, w = _create_dp_and_capture_warnings(
            output_path=self.CaO_DATA_DIR,
            skip_corrections=True,
        )
        assert not w
        assert len(dp.defect_dict) == 4
        self._check_DefectsParser(dp, skip_corrections=True)

        # v_Ca_+1 is an odd quartet state (S = 3/2, multiplicity = 4), and v_Ca_0 is a triplet states,
        # ISPIN = 2 calculations, manually checked. Looks like they could be band occupancies
        assert dp.defect_dict["v_Ca_+1"].degeneracy_factors["spin degeneracy"] == 4
        assert dp.defect_dict["v_Ca_0"].degeneracy_factors["spin degeneracy"] == 3

        # some hardcoded symmetry tests with default `symprec = 0.1` for relaxed structures:
        for name, vacancy_defect_entry in dp.defect_dict.items():
            print(
                name,
                vacancy_defect_entry.calculation_metadata["relaxed point symmetry"],
                vacancy_defect_entry.calculation_metadata["bulk site symmetry"],
            )
            assert vacancy_defect_entry.calculation_metadata["bulk site symmetry"] == "Oh"
        assert dp.defect_dict["v_Ca_+1"].calculation_metadata["relaxed point symmetry"] == "C2v"
        assert dp.defect_dict["v_Ca_0"].calculation_metadata["relaxed point symmetry"] == "C2v"
        assert dp.defect_dict["v_Ca_-1"].calculation_metadata["relaxed point symmetry"] == "C4v"
        assert dp.defect_dict["v_Ca_-2"].calculation_metadata["relaxed point symmetry"] == "Oh"

        thermo = dp.get_defect_thermodynamics()

        return thermo.plot()

    def test_BiOI_v_Bi_symmetry_determination(self):
        """
        Test parsing v_Bi_+1 from BiOI defect calculations, and confirming the
        correct point group symmetry of Cs is determined.
        """
        dp, w = _create_dp_and_capture_warnings(
            output_path=self.BiOI_DATA_DIR,
            skip_corrections=True,
        )
        assert not w
        assert len(dp.defect_dict) == 1
        self._check_DefectsParser(dp, skip_corrections=True)

        # some hardcoded symmetry tests with default `symprec = 0.1` for relaxed structures:
        assert dp.defect_dict["v_Bi_+1"].calculation_metadata["bulk site symmetry"] == "C4v"
        assert dp.defect_dict["v_Bi_+1"].calculation_metadata["relaxed point symmetry"] == "Cs"

        # test setting symprec during parsing
        dp, w = _create_dp_and_capture_warnings(
            output_path=self.BiOI_DATA_DIR,
            skip_corrections=True,
            symprec=0.01,
            parse_projected_eigen=False,  # just for fast testing, not recommended in general!
        )
        assert not w
        assert len(dp.defect_dict) == 1
        self._check_DefectsParser(dp, skip_corrections=True)

        # some hardcoded symmetry tests with default `symprec = 0.1` for relaxed structures:
        assert dp.defect_dict["v_Bi_+1"].calculation_metadata["bulk site symmetry"] == "C4v"
        assert dp.defect_dict["v_Bi_+1"].calculation_metadata["relaxed point symmetry"] == "C1"

        from doped.utils.symmetry import get_orientational_degeneracy

        assert get_orientational_degeneracy(dp.defect_dict["v_Bi_+1"]) == 4.0
        assert get_orientational_degeneracy(dp.defect_dict["v_Bi_+1"], symprec=0.01) == 8.0

    def test_shallow_defect_correction_warning_skipping(self):
        """
        Warnings about charge correction errors are skipped if the defects are
        not stable for any Fermi level in the gap (tested above in
        ``test_DefectsParser_corrections_errors_warning``) or if the defect is
        detected to be shallow (via ``pydefect`` eigenvalue analysis) and have
        a Fermi level stability region smaller than a given tolerance (given by
        the ``"shallow_charge_stability_tolerance"`` kwarg if set, otherwise
        the minimum of ``error_tolerance`` or 10% of the band gap value).

        This function tests the latter case.
        """

        def _check_shallow_O_Se_dp_w(dp, w, correction_warning=False, outcar_vr_mismatch=True):
            assert any("There are mismatching INCAR tags" in str(warn.message) for warn in w)
            assert any("('NKRED', 2, 1)" in str(warn.message) for warn in w)
            if outcar_vr_mismatch:  # warning about our artificially shifted vasprun energy:
                assert any(
                    "sub_1_O_on_Se_1/vasp_std:\nThe total energies of the provided (bulk) `OUTCAR` "
                    "(-381.559 eV), used to obtain the atomic core potentials for the eFNV correction, "
                    "and the `vasprun.xml` (-381.729eV, -363.622 eV; final energy & last electronic step "
                    "energy), used for" in str(warn.message)
                    for warn in w
                )
            # no charge correction warning by default, as charge correction error is only 6.36 meV here:
            assert any("Estimated error" in str(warn.message) for warn in w) == correction_warning
            assert (
                any("sub_1_O_on_Se_1: 6.36e-03 eV" in str(warn.message) for warn in w)
                == correction_warning
            )
            assert len(dp.defect_dict) == 2
            self._check_DefectsParser(dp)

        # Note that we have artificially modified the energy of ``sub_1_O_on_Se_1`` to be 0.17 eV lower,
        # so that it is found to be (just about) stable in the band gap for the purposes of this test
        dp, w = _create_dp_and_capture_warnings(
            output_path=self.shallow_O_Se_DATA_DIR, dielectric=self.Se_dielectric
        )

        _check_shallow_O_Se_dp_w(dp, w, correction_warning=False)
        thermo = dp.get_defect_thermodynamics()
        assert np.isclose(
            next(iter(thermo.transition_level_map["sub_1_O_on_Se"].keys())),
            0.00367,
            atol=1e-4,
        )
        assert np.isclose(
            thermo._get_in_gap_fermi_level_stability_window("sub_1_O_on_Se_1"),
            0.00367,
            atol=1e-4,
        )
        assert np.isclose(
            dp.defect_dict["sub_1_O_on_Se_1"].corrections_metadata["kumagai_charge_correction_error"],
            0.00636,
            atol=1e-4,
        )

        dp, w = _create_dp_and_capture_warnings(
            # error above tol but shallow with smaller stability window, no warning
            output_path=self.shallow_O_Se_DATA_DIR,
            dielectric=self.Se_dielectric,
            error_tolerance=0.005,
        )
        _check_shallow_O_Se_dp_w(dp, w, correction_warning=False)

        dp, w = _create_dp_and_capture_warnings(
            # error above tol, shallow but with larger stability window, warning
            output_path=self.shallow_O_Se_DATA_DIR,
            dielectric=self.Se_dielectric,
            error_tolerance=0.003,
        )
        _check_shallow_O_Se_dp_w(dp, w, correction_warning=True)

        dp, w = _create_dp_and_capture_warnings(
            # error > tol, shallow w/larger stability window, but `shallow_charge_stability_tolerance` set
            output_path=self.shallow_O_Se_DATA_DIR,
            dielectric=self.Se_dielectric,
            error_tolerance=0.003,
            shallow_charge_stability_tolerance=0.01,
        )
        _check_shallow_O_Se_dp_w(dp, w, correction_warning=False)

        # test parsing an incomplete vasprun and OUTCAR:
        vr_path = f"{self.shallow_O_Se_DATA_DIR}/sub_1_O_on_Se_1/vasp_std/vasprun.xml.gz"
        outcar_path = f"{self.shallow_O_Se_DATA_DIR}/sub_1_O_on_Se_1/vasp_std/OUTCAR.gz"
        with gzip.open(vr_path, "rt") as f:
            vr_lines = f.readlines()
        with gzip.open(outcar_path, "rt") as f:
            outcar_lines = f.readlines()

        shutil.move(vr_path, f"{vr_path}.bak")
        shutil.move(outcar_path, f"{outcar_path}.bak")
        with open(vr_path.strip(".gz"), "w") as out_file:
            out_file.writelines(vr_lines[:-2000])  # remove last 2000 lines
        with open(outcar_path.strip(".gz"), "w") as out_file:
            out_file.writelines(outcar_lines[:-500])

        dp, w = _create_dp_and_capture_warnings(
            output_path=self.shallow_O_Se_DATA_DIR,
            dielectric=self.Se_dielectric,
        )  # here the deletion of the last ionic step energy (which was artificially modified) means we
        # no longer have OUTCAR/vasprun.xml energy mismatch:
        _check_shallow_O_Se_dp_w(dp, w, correction_warning=False, outcar_vr_mismatch=False)

    def test_auto_charge_determination(self):
        """
        Test that the defect charge is correctly auto-determined.
        """
        defect_path = f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl"
        # test warning when specifying the wrong charge:
        with warnings.catch_warnings(record=True) as w:
            parsed_v_cd_m1 = defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=self.CdTe_dielectric,
                charge_state=-1,
                parse_projected_eigen=False,  # just for fast testing, not recommended in general!
            )
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert (
                "Auto-determined system charge q=-2 does not match specified charge q=-1. Will continue "
                "with specified charge_state, but beware!" in str(w[-1].message)
            )
            assert np.isclose(
                parsed_v_cd_m1.corrections["freysoldt_charge_correction"], 0.261, atol=1e-3
            )  # slightly higher atol, due to LOCPOT sub-sampling for file compression on repo

        # test YTOS, has trickier POTCAR symbols with  Y_sv, Ti, S, O
        ytos_F_O_1 = defect_entry_from_paths(  # with corrections this time
            f"{self.YTOS_EXAMPLE_DIR}/F_O_1",
            f"{self.YTOS_EXAMPLE_DIR}/Bulk",
            self.ytos_dielectric,
        )
        ytos_F_O_1_explicit = defect_entry_from_paths(  # with corrections this time
            f"{self.YTOS_EXAMPLE_DIR}/F_O_1",
            f"{self.YTOS_EXAMPLE_DIR}/Bulk",
            self.ytos_dielectric,
            charge_state=1,
        )
        assert ytos_F_O_1.charge_state == ytos_F_O_1_explicit.charge_state == 1
        assert ytos_F_O_1.get_ediff() == ytos_F_O_1_explicit.get_ediff()
        assert np.isclose(ytos_F_O_1.get_ediff(), 0.04176070572680146, atol=1e-3)  # corrected energy
        correction_dict = {
            "kumagai_charge_correction": 0.12699488572686776,
        }
        for correction_name, correction_energy in correction_dict.items():
            assert np.isclose(ytos_F_O_1.corrections[correction_name], correction_energy, atol=1e-3)
        # assert auto-determined interstitial site is correct
        assert np.isclose(
            ytos_F_O_1.defect_supercell_site.distance_and_image_from_frac_coords([0, 0, 0])[0],
            0.0,
            atol=1e-2,
        )

    def test_auto_charge_correction_behaviour(self):
        """
        Test skipping of charge corrections and warnings.

        Here we have mixed and matched `defect_entry_from_paths` and
        `DefectParser.from_paths()` as the results should be the same.

        Note that we have manually edited the LOCPOTs in the examples
        folders to reduce space, using the following code:

        from pymatgen.io.vasp.outputs import Locpot
        import numpy as np

        filename = "..."
        locpot = Locpot.from_file(filename)
        locpot.is_spin_polarized = False

        def blockwise_average_ND(a, factors):
           # from https://stackoverflow.com/questions/37532184/
           # downsize-3d-matrix-by-averaging-in-numpy-or-alike/73078468#73078468
            # `a` is the N-dim input array
            # `factors` is the blocksize on which averaging is to be performed

            factors = np.asanyarray(factors)
            sh = np.column_stack([a.shape//factors, factors]).ravel()
            b = a.reshape(sh).mean(tuple(range(1, 2*a.ndim, 2)))

            return b

        locpot.data["total"] = blockwise_average_ND(locpot.data["total"], [2, 2, 2])
        locpot.dim = [locpot.dim[0] // 2, locpot.dim[1] // 2, locpot.dim[2] // 2]
        locpot.write_file(filename)
        """
        defect_path = f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl"
        fake_aniso_dielectric = [1, 2, 3]

        with warnings.catch_warnings(record=True) as w:
            parsed_v_cd_m2_fake_aniso_dp = DefectParser.from_paths(
                defect_path=defect_path,
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=fake_aniso_dielectric,
                parse_projected_eigen=False,  # just for fast testing, not recommended in general!
            )
        print([str(warn.message) for warn in w])  # for debugging
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert (
            "An anisotropic dielectric constant was supplied, but `OUTCAR` files (needed to compute "
            "the _anisotropic_ Kumagai eFNV charge correction) are missing from the defect or bulk "
            "folder.\n`LOCPOT` files were found in both defect & bulk folders, and so the "
            "Freysoldt (FNV) charge correction developed for _isotropic_ materials will be applied "
            "here, which corresponds to using the effective isotropic average of the supplied "
            "anisotropic dielectric. This could lead to significant errors for very anisotropic "
            "systems and/or relatively small supercells!" in str(w[-1].message)
        )
        assert all(
            i in parsed_v_cd_m2_fake_aniso_dp.__repr__()
            for i in [
                "doped DefectParser for bulk composition CdTe. ",
                "Available attributes",
                "defect_entry",
                "error_tolerance",
                "Available methods",
                "load_eFNV_data",
            ]
        )
        parsed_v_cd_m2_fake_aniso = parsed_v_cd_m2_fake_aniso_dp.defect_entry

        assert np.isclose(
            parsed_v_cd_m2_fake_aniso.get_ediff() - sum(parsed_v_cd_m2_fake_aniso.corrections.values()),
            7.661,
            atol=3e-3,
        )  # uncorrected energy
        assert np.isclose(parsed_v_cd_m2_fake_aniso.get_ediff(), 10.379714081555262, atol=1e-3)

        # test no warnings when skip_corrections is True
        with warnings.catch_warnings(record=True) as w:
            parsed_v_cd_m2_fake_aniso = defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=fake_aniso_dielectric,
                skip_corrections=True,
                charge_state=-2,
            )
            assert len(w) == 0

        assert np.isclose(
            parsed_v_cd_m2_fake_aniso.get_ediff() - sum(parsed_v_cd_m2_fake_aniso.corrections.values()),
            7.661,
            atol=3e-3,
        )  # uncorrected energy
        assert np.isclose(parsed_v_cd_m2_fake_aniso.get_ediff(), 7.661, atol=1e-3)
        assert parsed_v_cd_m2_fake_aniso.corrections == {}

        # test fake anisotropic dielectric with Int_Te_3_2, which has multiple OUTCARs:
        with warnings.catch_warnings(record=True) as w:
            parsed_int_Te_2_fake_aniso = DefectParser.from_paths(
                defect_path=f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl",
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=fake_aniso_dielectric,
                charge_state=2,  # test manually specifying charge state
                parse_projected_eigen=False,  # just for fast testing, not recommended in general!
            ).defect_entry
            assert (
                f"Multiple `OUTCAR` files found in defect directory: "
                f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl. Using OUTCAR.gz to parse core levels and "
                f"compute the Kumagai (eFNV) image charge correction." in str(w[0].message)
            )
            assert (
                f"Estimated error in the Kumagai (eFNV) charge correction for defect "
                f"{parsed_int_Te_2_fake_aniso.name} is 0.157 eV (i.e. which is greater than the "
                f"`error_tolerance`: 0.050 eV). You may want to check the accuracy of the correction "
                f"by plotting the site potential differences (using "
                f"`defect_entry.get_kumagai_correction()` with `plot=True`). Large errors are often due "
                f"to unstable or shallow defect charge states (which can't be accurately modelled with "
                f"the supercell approach; see "
                f"https://doped.readthedocs.io/en/latest/Tips.html#perturbed-host-states-shallow-defects"
                f"). If this error is not acceptable, you may need to use a larger supercell for more "
                f"accurate energies." in str(w[1].message)
            )

        assert np.isclose(
            parsed_int_Te_2_fake_aniso.get_ediff() - sum(parsed_int_Te_2_fake_aniso.corrections.values()),
            -7.105,
            atol=3e-3,
        )  # uncorrected energy
        assert np.isclose(parsed_int_Te_2_fake_aniso.get_ediff(), -4.991240009587045, atol=1e-3)

        # test isotropic dielectric but only OUTCAR present:
        with warnings.catch_warnings(record=True) as w:
            parsed_int_Te_2 = defect_entry_from_paths(
                defect_path=f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl",
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=self.CdTe_dielectric,
                charge_state=2,
            )
        assert len(w) == 1  # no charge correction warning with iso dielectric, parsing from OUTCARs,
        # but multiple OUTCARs present -> warning
        assert np.isclose(parsed_int_Te_2.get_ediff(), -6.2009, atol=1e-3)

        # test warning when only OUTCAR present but no core level info (ICORELEVEL != 0)
        shutil.move(
            f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/OUTCAR.gz",
            f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/hidden_otcr.gz",
        )

        with warnings.catch_warnings(record=True) as w:
            parsed_int_Te_2_fake_aniso = self._check_no_icorelevel_warning_int_te(
                fake_aniso_dielectric,
                w,
                1,
                "-> Charge corrections will not be applied for this defect.",
            )
        assert np.isclose(
            parsed_int_Te_2_fake_aniso.get_ediff() - sum(parsed_int_Te_2_fake_aniso.corrections.values()),
            -7.105,
            atol=3e-3,
        )  # uncorrected energy
        assert np.isclose(parsed_int_Te_2_fake_aniso.get_ediff(), -7.105, atol=1e-3)

        # test warning when no core level info in OUTCAR (ICORELEVEL != 0), but LOCPOT
        # files present, but anisotropic dielectric:
        shutil.copyfile(
            f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl/LOCPOT.gz",
            f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/LOCPOT.gz",
        )

        with warnings.catch_warnings(record=True) as w:
            parsed_int_Te_2_fake_aniso = self._check_no_icorelevel_warning_int_te(
                fake_aniso_dielectric,
                w,
                2,
                "`LOCPOT` files were found in both defect & bulk folders, and so the Freysoldt (FNV) "
                "charge correction developed for _isotropic_ materials will be applied here, "
                "which corresponds to using the effective isotropic average of the supplied anisotropic "
                "dielectric. This could lead to significant errors for very anisotropic systems and/or "
                "relatively small supercells!",
            )

        assert np.isclose(
            parsed_int_Te_2_fake_aniso.get_ediff() - sum(parsed_int_Te_2_fake_aniso.corrections.values()),
            -7.105,
            atol=3e-3,
        )  # uncorrected energy
        assert np.isclose(
            parsed_int_Te_2_fake_aniso.get_ediff(), -4.7620, atol=2e-2
        )  # -4.734 with old voronoi frac coords, expanded atol now to account for LOCPOT sub-sampling
        # compression

        if_present_rm(f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/LOCPOT.gz")

        # rename files back to original:
        shutil.move(
            f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/hidden_otcr.gz",
            f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/OUTCAR.gz",
        )

        # test warning when no OUTCAR or LOCPOT file found:
        defect_path = f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl"
        shutil.move(
            f"{defect_path}/LOCPOT.gz",
            f"{defect_path}/hidden_lcpt.gz",
        )
        with warnings.catch_warnings(record=True) as w:
            parsed_v_cd_m2 = DefectParser.from_paths(
                defect_path=defect_path,
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=self.CdTe_dielectric,
            ).defect_entry
            assert len(w) == 1
            assert all(issubclass(warning.category, UserWarning) for warning in w)
            assert (
                "`LOCPOT` or `OUTCAR` files are missing from the defect or bulk folder. These are needed "
                "to perform the finite-size charge corrections. Charge corrections will not be applied "
                "for this defect." in str(w[0].message)
            )

        assert parsed_v_cd_m2.charge_state == -2
        assert np.isclose(
            parsed_v_cd_m2.get_ediff() - sum(parsed_v_cd_m2.corrections.values()), 7.661, atol=3e-3
        )  # uncorrected energy
        assert np.isclose(parsed_v_cd_m2.get_ediff(), 7.661, atol=1e-3)
        assert parsed_v_cd_m2.corrections == {}

        # move LOCPOT back to original:
        shutil.move(f"{defect_path}/hidden_lcpt.gz", f"{defect_path}/LOCPOT.gz")

        # test no warning when no OUTCAR or LOCPOT file found, but charge is zero:
        defect_path = f"{self.CdTe_EXAMPLE_DIR}/v_Cd_0/vasp_ncl"  # no LOCPOT/OUTCAR

        with warnings.catch_warnings(record=True) as w:
            parsed_v_cd_0 = defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=self.CdTe_dielectric,
                parse_projected_eigen=False,  # just for fast testing, not recommended in general!
            )
            assert len(w) == 0

        assert np.isclose(
            parsed_v_cd_0.get_ediff() - sum(parsed_v_cd_0.corrections.values()), 4.166, atol=3e-3
        )  # uncorrected energy
        assert np.isclose(parsed_v_cd_0.get_ediff(), 4.166, atol=1e-3)

    def _check_no_icorelevel_warning_int_te(self, dielectric, warnings, num_warnings, action):
        print(
            f"Running _check_no_icorelevel_warning_int_te with dielectric {dielectric}, expecting "
            f"{num_warnings} warnings and action: {action}"
        )  # for debugging
        result = defect_entry_from_paths(
            defect_path=f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl",
            bulk_path=self.CdTe_BULK_DATA_DIR,
            dielectric=dielectric,
            charge_state=2,
        )
        print([warn.message for warn in warnings])  # for debugging
        assert len(warnings) == num_warnings
        assert all(issubclass(warning.category, UserWarning) for warning in warnings)
        assert (  # different warning start depending on whether isotropic or anisotropic dielectric
            f"in the defect or bulk folder were unable to be parsed, giving the following error message:\n"
            f"Unable to parse atomic core potentials from defect `OUTCAR` at "
            f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/OUTCAR_no_core_levels.gz. This can happen if "
            f"`ICORELEVEL` was not set to 0 (= default) in the `INCAR`, the calculation was finished "
            f"prematurely with a `STOPCAR`, or the calculation crashed. The Kumagai (eFNV) charge "
            f"correction cannot be computed without this data!\n{action}" in str(warnings[0].message)
        )

        return result

    def _parse_Int_Te_3_2_and_count_warnings(self, fake_aniso_dielectric, w, num_warnings):
        print(
            f"Running _parse_Int_Te_3_2_and_count_warnings with dielectric {fake_aniso_dielectric}, "
            f"expecting {num_warnings} warnings"
        )  # for debugging
        defect_entry_from_paths(
            defect_path=f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl",
            bulk_path=self.CdTe_BULK_DATA_DIR,
            dielectric=fake_aniso_dielectric,
            charge_state=2,
            parse_projected_eigen=False,  # just for fast testing, not recommended in general!
        )
        print([warn.message for warn in w])  # for debugging
        assert len(w) == num_warnings
        assert all(issubclass(warning.category, UserWarning) for warning in w)

    def test_multiple_outcars(self):
        shutil.copyfile(
            f"{self.CdTe_BULK_DATA_DIR}/OUTCAR.gz",
            f"{self.CdTe_BULK_DATA_DIR}/another_OUTCAR.gz",
        )
        fake_aniso_dielectric = [1, 2, 3]
        with warnings.catch_warnings(record=True) as w:
            self._parse_Int_Te_3_2_and_count_warnings(fake_aniso_dielectric, w, 3)
            assert (
                f"Multiple `OUTCAR` files found in bulk directory: {self.CdTe_BULK_DATA_DIR}. Using "
                f"OUTCAR.gz to parse core levels and compute the Kumagai (eFNV) image charge "
                f"correction." in str(w[0].message)
            )
            assert (
                f"Multiple `OUTCAR` files found in defect directory: "
                f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl. Using "
                f"OUTCAR.gz to parse core levels and compute the Kumagai (eFNV) image charge "
                f"correction." in str(w[1].message)
            )
            # other warnings is charge correction error warning, already tested

        with warnings.catch_warnings(record=True) as w:
            self._parse_Int_Te_3_2_and_count_warnings(fake_aniso_dielectric, w, 3)

    def test_multiple_locpots(self):
        defect_path = f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl"

        shutil.copyfile(f"{defect_path}/LOCPOT.gz", f"{defect_path}/another_LOCPOT.gz")
        shutil.copyfile(
            f"{self.CdTe_BULK_DATA_DIR}/LOCPOT.gz",
            f"{self.CdTe_BULK_DATA_DIR}/another_LOCPOT.gz",
        )

        with warnings.catch_warnings(record=True) as w:
            defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=self.CdTe_dielectric,
                charge_state=-2,
                parse_projected_eigen=False,  # just for fast testing, not recommended in general!
            )
            assert len(w) == 2  # multiple LOCPOTs (both defect and bulk)
            assert all(issubclass(warning.category, UserWarning) for warning in w)
            assert (
                f"Multiple `LOCPOT` files found in bulk directory: {self.CdTe_BULK_DATA_DIR}. Using "
                f"LOCPOT.gz to parse the electrostatic potential and compute the Freysoldt (FNV) charge "
                f"correction." in str(w[0].message)
            )
            assert (
                f"Multiple `LOCPOT` files found in defect directory: {defect_path}. Using LOCPOT.gz to "
                f"parse the electrostatic potential and compute the Freysoldt (FNV) charge correction."
                in str(w[1].message)
            )

    def test_multiple_vaspruns(self):
        defect_path = f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl"

        shutil.copyfile(f"{defect_path}/vasprun.xml.gz", f"{defect_path}/another_vasprun.xml.gz")
        shutil.copyfile(
            f"{self.CdTe_BULK_DATA_DIR}/vasprun.xml.gz",
            f"{self.CdTe_BULK_DATA_DIR}/another_vasprun.xml.gz",
        )

        with warnings.catch_warnings(record=True) as w:
            defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=self.CdTe_dielectric,
                charge_state=-2,  # test manually specifying charge state
                parse_projected_eigen=False,  # just for fast testing, not recommended in general!
            )
            assert len(w) == 2  # multiple `vasprun.xml`s (both defect and bulk)
            assert all(issubclass(warning.category, UserWarning) for warning in w)
            assert (
                f"Multiple `vasprun.xml` files found in bulk directory: {self.CdTe_BULK_DATA_DIR}. Using "
                f"vasprun.xml.gz to parse the calculation energy and metadata." in str(w[0].message)
            )
            assert (
                f"Multiple `vasprun.xml` files found in defect directory: {defect_path}. Using "
                f"vasprun.xml.gz to parse the calculation energy and metadata." in str(w[1].message)
            )

    def test_dielectric_initialisation(self):
        """
        Test that dielectric can be supplied as float or int or 3x1 array/list
        or 3x3 array/list.

        This test currently takes about 5 minutes, which is mainly due to slow-
        ish parsing of LOCPOT correction files. If we wanted to speed up, could
        refactor these tests to use an eFNV-corrected defect!
        """
        defect_path = f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl"
        # get correct Freysoldt correction energy:
        parsed_v_cd_m2 = defect_entry_from_paths(  # defect charge determined automatically
            defect_path=defect_path,
            bulk_path=self.CdTe_BULK_DATA_DIR,
            dielectric=self.CdTe_dielectric,
            charge_state=-2,
            parse_projected_eigen=False,  # just for fast testing, not recommended in general!
        )

        # Check that the correct Freysoldt correction is applied
        correct_correction_dict = {
            "freysoldt_charge_correction": 0.7376460317828045,
        }
        for correction_name, correction_energy in correct_correction_dict.items():
            assert np.isclose(
                parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                atol=2e-3,  # slightly higher atol due to LOCPOT sub-sampling for file compression
            )

        # test float
        new_parsed_v_cd_m2 = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=self.CdTe_BULK_DATA_DIR,
            dielectric=9.13,
            parse_projected_eigen=False,  # just for fast testing, not recommended in general!
        )
        for correction_name, correction_energy in correct_correction_dict.items():
            assert np.isclose(
                new_parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                atol=2e-3,  # slightly higher atol due to LOCPOT sub-sampling for file compression
            )

        # test int
        new_parsed_v_cd_m2 = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=self.CdTe_BULK_DATA_DIR,
            dielectric=9,
            charge_state=-2,
            parse_projected_eigen=False,  # just for fast testing, not recommended in general!
        )
        for correction_name, correction_energy in correct_correction_dict.items():
            assert np.isclose(
                new_parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                atol=0.1,  # now slightly off because using int()
            )

        # test 3x1 array
        new_parsed_v_cd_m2 = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=self.CdTe_BULK_DATA_DIR,
            dielectric=np.array([9.13, 9.13, 9.13]),
            parse_projected_eigen=False,  # just for fast testing, not recommended in general!
        )
        for correction_name, correction_energy in correct_correction_dict.items():
            assert np.isclose(
                new_parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                atol=2e-3,  # slightly higher atol due to LOCPOT sub-sampling for file compression
            )

        # test 3x1 list
        new_parsed_v_cd_m2 = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=self.CdTe_BULK_DATA_DIR,
            dielectric=[9.13, 9.13, 9.13],
            charge_state=-2,
            parse_projected_eigen=False,  # just for fast testing, not recommended in general!
        )
        for correction_name, correction_energy in correct_correction_dict.items():
            assert np.isclose(
                new_parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                atol=2e-3,  # slightly higher atol due to LOCPOT sub-sampling for file compression
            )

        # test 3x3 array
        new_parsed_v_cd_m2 = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=self.CdTe_BULK_DATA_DIR,
            dielectric=self.CdTe_dielectric,
            parse_projected_eigen=False,  # just for fast testing, not recommended in general!
        )
        for correction_name, correction_energy in correct_correction_dict.items():
            assert np.isclose(
                new_parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                atol=2e-3,  # slightly higher atol due to LOCPOT sub-sampling for file compression
            )

        # test 3x3 list
        new_parsed_v_cd_m2 = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=self.CdTe_BULK_DATA_DIR,
            dielectric=self.CdTe_dielectric.tolist(),
            charge_state=-2,
            parse_projected_eigen=False,  # just for fast testing, not recommended in general!
        )
        for correction_name, correction_energy in correct_correction_dict.items():
            assert np.isclose(
                new_parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                atol=2e-3,  # slightly higher atol due to LOCPOT sub-sampling for file compression
            )

    def test_vacancy_parsing_and_freysoldt(self):
        """
        Test parsing of Cd vacancy calculations and correct Freysoldt
        correction calculated.
        """
        parsed_vac_Cd_dict = {}

        for i in os.listdir(self.CdTe_EXAMPLE_DIR):
            # loop folders and parse those with "v_Cd" in name:
            if os.path.isdir(f"{self.CdTe_EXAMPLE_DIR}/{i}") and "v_Cd" in i and "example" not in i:
                defect_path = f"{self.CdTe_EXAMPLE_DIR}/{i}/vasp_ncl"
                int(i[-2:].replace("_", ""))
                parsed_vac_Cd_dict[i] = defect_entry_from_paths(
                    defect_path=defect_path,
                    bulk_path=self.CdTe_BULK_DATA_DIR,
                    dielectric=self.CdTe_dielectric,
                )  # Keep dictionary of parsed defect entries

        assert len(parsed_vac_Cd_dict) == 3
        assert all(f"v_Cd_{i}" in parsed_vac_Cd_dict for i in [0, -1, -2])
        # Check that the correct Freysoldt correction is applied
        for name, energy, correction_dict in [
            (
                "v_Cd_0",
                4.166,
                {},
            ),
            (
                "v_Cd_-1",
                6.355,
                {
                    "freysoldt_charge_correction": 0.22517150393292082,
                },
            ),
            (
                "v_Cd_-2",
                8.398,
                {
                    "freysoldt_charge_correction": 0.7376460317828045,
                },
            ),
        ]:
            assert np.isclose(parsed_vac_Cd_dict[name].get_ediff(), energy, atol=2e-3)
            for correction_name, correction_energy in correction_dict.items():
                assert np.isclose(
                    parsed_vac_Cd_dict[name].corrections[correction_name],
                    correction_energy,
                    atol=2e-3,  # slightly higher atol due to LOCPOT sub-sampling for file compression
                )

            # assert auto-determined vacancy site is correct
            # should be: PeriodicSite: Cd (6.5434, 6.5434, 6.5434) [0.5000, 0.5000, 0.5000]
            assert np.allclose(
                parsed_vac_Cd_dict[name].defect_supercell_site.frac_coords,
                [0.5, 0.5, 0.5] if name == "v_Cd_0" else [0, 0, 0],
            )

    def test_interstitial_parsing_and_kumagai(self):
        """
        Test parsing of Te (split-)interstitial and Kumagai-Oba (eFNV)
        correction.
        """
        with patch("builtins.print") as mock_print:
            te_i_2_ent = defect_entry_from_paths(
                defect_path=f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl",
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=self.CdTe_dielectric,
                charge_state=+2,  # test manually specifying charge state
                parse_projected_eigen=False,  # just for fast testing, not recommended in general!
            )

        self._check_defect_entry_corrections(te_i_2_ent, -6.2009, 0.9038318161163628)
        # assert auto-determined interstitial site is correct
        # initial position is: PeriodicSite: Te (12.2688, 12.2688, 8.9972) [0.9375, 0.9375, 0.6875]
        assert np.allclose(te_i_2_ent.defect_supercell_site.frac_coords, [0.834511, 0.943944, 0.69776])

        # run again to check parsing of previous Voronoi sites
        with patch("builtins.print") as mock_print:
            te_i_2_ent = defect_entry_from_paths(
                defect_path=f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl",
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=self.CdTe_dielectric,
                charge_state=+2,  # test manually specifying charge state
                parse_projected_eigen=False,  # just for fast testing, not recommended in general!
            )

        mock_print.assert_not_called()

    def test_substitution_parsing_and_kumagai(self):
        """
        Test parsing of Te_Cd_1 and Kumagai-Oba (eFNV) correction.
        """
        for i in os.listdir(self.CdTe_EXAMPLE_DIR):
            if "Te_Cd" in i:  # loop folders and parse those with "Te_Cd" in name
                defect_path = f"{self.CdTe_EXAMPLE_DIR}/{i}/vasp_ncl"
                defect_charge = int(i[-2:].replace("_", ""))
                te_cd_1_ent = defect_entry_from_paths(
                    defect_path=defect_path,
                    bulk_path=self.CdTe_BULK_DATA_DIR,
                    dielectric=self.CdTe_dielectric,
                    charge_state=defect_charge,
                    parse_projected_eigen=False,  # just for fast testing, not recommended in general!
                )

        self._check_defect_entry_corrections(te_cd_1_ent, -2.6676, 0.23840982963691623)
        # assert auto-determined substitution site is correct
        # should be: PeriodicSite: Te (6.5434, 6.5434, 6.5434) [0.5000, 0.5000, 0.5000]
        assert np.allclose(te_cd_1_ent.defect_supercell_site.frac_coords, [0.475139, 0.475137, 0.524856])

    def test_extrinsic_interstitial_defect_ID(self):
        """
        Test parsing of extrinsic F in YTOS interstitial.
        """
        bulk_sc_structure = Structure.from_file(f"{self.YTOS_EXAMPLE_DIR}/Bulk/POSCAR")
        initial_defect_structure = Structure.from_file(f"{self.YTOS_EXAMPLE_DIR}/Int_F_-1/Relaxed_CONTCAR")
        (def_type, comp_diff) = get_defect_type_and_composition_diff(
            bulk_sc_structure, initial_defect_structure
        )
        assert def_type == "interstitial"
        assert comp_diff == {"F": 1}
        (
            def_type,
            bulk_site_idx,
            defect_site_idx,
            unrelaxed_defect_structure,
        ) = get_defect_type_site_idxs_and_unrelaxed_structure(bulk_sc_structure, initial_defect_structure)
        assert bulk_site_idx is None
        assert def_type == "interstitial"
        assert defect_site_idx == len(unrelaxed_defect_structure) - 1

        # assert auto-determined interstitial site is correct
        assert np.isclose(
            unrelaxed_defect_structure[defect_site_idx].distance_and_image_from_frac_coords(
                [-0.0005726049122470, -0.0001544430438804, 0.47800736578014720]
            )[0],
            0.0,
            atol=1e-2,
        )  # approx match, not exact because relaxed bulk supercell

    def test_extrinsic_substitution_defect_ID(self):
        """
        Test parsing of extrinsic U_on_Cd in CdTe.
        """
        bulk_sc_structure = Structure.from_file(
            f"{self.CdTe_EXAMPLE_DIR}/CdTe_bulk/CdTe_bulk_supercell_POSCAR"
        )
        initial_defect_structure = Structure.from_file(f"{self.CdTe_EXAMPLE_DIR}/U_on_Cd_POSCAR")
        (
            def_type,
            comp_diff,
        ) = get_defect_type_and_composition_diff(bulk_sc_structure, initial_defect_structure)
        assert def_type == "substitution"
        assert comp_diff == {"Cd": -1, "U": 1}
        (
            def_type,
            bulk_site_idx,
            defect_site_idx,
            unrelaxed_defect_structure,
        ) = get_defect_type_site_idxs_and_unrelaxed_structure(bulk_sc_structure, initial_defect_structure)
        assert def_type == "substitution"
        assert bulk_site_idx == 0
        assert defect_site_idx == 63  # last site in structure

        # assert auto-determined substitution site is correct (exact match because perfect supercell):
        assert np.array_equal(unrelaxed_defect_structure[defect_site_idx].frac_coords, [0.00, 0.00, 0.00])

    @custom_mpl_image_compare("YTOS_Int_F_-1_eigenvalue_plot_ISPIN_1.png")
    def test_extrinsic_interstitial_parsing_and_kumagai(self):
        """
        Test parsing of extrinsic F in YTOS interstitial and Kumagai-Oba (eFNV)
        correction.

        Also tests output of eigenvalue analysis for a defect & bulk
        calculated with ``ISPIN = 1`` (all previous cases are either ``ISPIN = 2``
        or with SOC).
        """
        defect_path = f"{self.YTOS_EXAMPLE_DIR}/Int_F_-1/"

        # parse with no explicitly-set-charge:
        with warnings.catch_warnings(record=True) as w:
            int_F_minus1_ent = defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk/",
                dielectric=self.ytos_dielectric,
                parse_projected_eigen=False,  # just for fast testing, not recommended in general!
            )
        assert not [warning for warning in w if issubclass(warning.category, UserWarning)]
        bes, eig_fig = int_F_minus1_ent.get_eigenvalue_analysis()
        assert not any(
            [
                bes.has_acceptor_phs,
                bes.has_donor_phs,
                bes.has_occupied_localized_state,
                bes.has_unoccupied_localized_state,
                bes.is_shallow,
            ]
        )

        correction_dict = self._check_defect_entry_corrections(
            int_F_minus1_ent, 0.7478967131628451, -0.0036182568370900017
        )
        # assert auto-determined interstitial site is correct
        assert np.isclose(
            int_F_minus1_ent.defect_supercell_site.distance_and_image_from_frac_coords(
                [-0.0005726049122470, -0.0001544430438804, 0.4780073657801472]
            )[
                0
            ],  # relaxed site
            0.0,
            atol=1e-2,
        )  # approx match, not exact because relaxed bulk supercell

        # test error_tolerance setting:
        with warnings.catch_warnings(record=True) as w:
            int_F_minus1_ent = defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk/",
                dielectric=self.ytos_dielectric,
                error_tolerance=0.001,
            )
        assert (
            f"Estimated error in the Kumagai (eFNV) charge correction for defect "
            f"{int_F_minus1_ent.name} is 2.58e-03 eV (i.e. which is greater than the `error_tolerance`: "
            f"1.00e-03 eV). You may want to check the accuracy of the correction by plotting the site "
            f"potential differences (using `defect_entry.get_kumagai_correction()` with "
            f"`plot=True`). Large errors are often due to unstable or shallow defect charge states ("
            f"which can't be accurately modelled with the supercell approach; see "
            f"https://doped.readthedocs.io/en/latest/Tips.html#perturbed-host-states-shallow-defects). If "
            f"this error is not acceptable, you may need to use a larger supercell for more accurate "
            f"energies." in str(w[0].message)
        )

        with warnings.catch_warnings(record=True) as w:
            int_F_minus1_ent.get_kumagai_correction()  # default error tolerance, no warning
        assert not [warning for warning in w if issubclass(warning.category, UserWarning)]

        with warnings.catch_warnings(record=True) as w:
            int_F_minus1_ent.get_kumagai_correction(error_tolerance=0.001)
        assert "Estimated error in the Kumagai (eFNV)" in str(w[0].message)

        # test returning correction error:
        corr, corr_error = int_F_minus1_ent.get_kumagai_correction(return_correction_error=True)
        assert np.isclose(corr.correction_energy, correction_dict["kumagai_charge_correction"], atol=1e-3)
        assert np.isclose(corr_error, 0.003, atol=1e-3)
        assert np.isclose(
            int_F_minus1_ent.corrections_metadata["kumagai_charge_correction_error"], 0.003, atol=1e-3
        )

        # test returning correction error with plot:
        corr, fig, corr_error = int_F_minus1_ent.get_kumagai_correction(
            return_correction_error=True, plot=True
        )
        assert np.isclose(corr.correction_energy, correction_dict["kumagai_charge_correction"], atol=1e-3)
        assert np.isclose(corr_error, 0.003, atol=1e-3)
        assert np.isclose(
            int_F_minus1_ent.corrections_metadata["kumagai_charge_correction_error"], 0.003, atol=1e-3
        )

        # test just correction returned with plot = False and return_correction_error = False:
        corr = int_F_minus1_ent.get_kumagai_correction()
        assert np.isclose(corr.correction_energy, correction_dict["kumagai_charge_correction"], atol=1e-3)

        # test symmetry determination (periodicity breaking does not affect F_i):
        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            relaxed_defect_name = get_defect_name_from_entry(int_F_minus1_ent)
            print([warn.message for warn in w])  # for debugging
            assert not w  # this supercell is not periodicity breaking
        # F_i conventional structure interstitial coords here: [0, 0, 0.48467759]
        assert relaxed_defect_name == "F_i_C4v_O2.57"
        assert get_defect_name_from_entry(int_F_minus1_ent, relaxed=False) == "F_i_C4v_O2.57"

        return eig_fig  # test eigenvalue plot for ISPIN = 1 case

    def _check_defect_entry_corrections(self, defect_entry, ediff, correction):
        assert np.isclose(defect_entry.get_ediff(), ediff, atol=0.001)
        assert np.isclose(
            defect_entry.get_ediff() - sum(defect_entry.corrections.values()),
            ediff - correction,
            atol=0.003,
        )
        correction_dict = {"kumagai_charge_correction": correction}
        for correction_name, correction_energy in correction_dict.items():
            assert np.isclose(defect_entry.corrections[correction_name], correction_energy, atol=0.001)
        return correction_dict

    def test_extrinsic_substitution_parsing_and_freysoldt_and_kumagai(self):
        """
        Test parsing of extrinsic F-on-O substitution in YTOS, w/Kumagai-Oba
        (eFNV) and Freysoldt (FNV) corrections.
        """
        # first using Freysoldt (FNV) correction
        defect_path = f"{self.YTOS_EXAMPLE_DIR}/F_O_1/"
        # hide OUTCAR file:
        shutil.move(f"{defect_path}/OUTCAR.gz", f"{defect_path}/hidden_otcr.gz")

        # parse with no explicitly-set-charge:
        with warnings.catch_warnings(record=True) as w:
            F_O_1_ent = defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk/",
                dielectric=self.ytos_dielectric,
                parse_projected_eigen=False,  # just for fast testing, not recommended in general!
            )  # check no correction error warning with default tolerance:
        assert len([warning for warning in w if issubclass(warning.category, UserWarning)]) == 1
        assert "An anisotropic dielectric constant was supplied, but `OUTCAR`" in str(w[0].message)

        # test error_tolerance setting:
        with warnings.catch_warnings(record=True) as w:
            F_O_1_ent = defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk/",
                dielectric=self.ytos_dielectric,
                error_tolerance=0.00001,
                parse_projected_eigen=False,  # just for fast testing, not recommended in general!
            )  # check no correction error warning with default tolerance:
        print([str(warn.message) for warn in w])
        assert any(  # break up warning message to allow slightly difference numbers after 3.XXe-04 eV:
            f"Estimated error in the Freysoldt (FNV) charge correction for defect {F_O_1_ent.name} is 3"
            in str(warning.message)
            for warning in w
        )
        assert any(
            "e-04 eV (i.e. which is greater than the `error_tolerance`: 1.00e-05 eV). You may want "
            "to check the accuracy of the correction by plotting the site potential differences (using "
            "`defect_entry.get_freysoldt_correction()` with `plot=True`). Large errors are often due "
            "to unstable or shallow defect charge states (which can't be accurately modelled with "
            "the supercell approach; see "
            "https://doped.readthedocs.io/en/latest/Tips.html#perturbed-host-states-shallow-defects). "
            "If this error is not acceptable, you may need to use a larger supercell for more "
            "accurate energies." in str(warning.message)
            for warning in w
        )

        with warnings.catch_warnings(record=True) as w:
            F_O_1_ent.get_freysoldt_correction()  # default error tolerance, no warning
        assert not [warning for warning in w if issubclass(warning.category, UserWarning)]

        with warnings.catch_warnings(record=True) as w:
            F_O_1_ent.get_freysoldt_correction(error_tolerance=0.00001)
        assert "Estimated error in the Freysoldt (FNV)" in str(w[0].message)

        # test returning correction error:
        corr, corr_error = F_O_1_ent.get_freysoldt_correction(return_correction_error=True)
        assert np.isclose(corr.correction_energy, 0.11670254204631794, atol=1e-3)
        assert np.isclose(corr_error, 0.000, atol=1e-3)
        assert np.isclose(
            F_O_1_ent.corrections_metadata["freysoldt_charge_correction_error"], 0.000, atol=1e-3
        )

        # test returning correction error with plot:
        corr, fig, corr_error = F_O_1_ent.get_freysoldt_correction(return_correction_error=True, plot=True)
        assert np.isclose(corr.correction_energy, 0.11670254204631794, atol=1e-3)
        assert np.isclose(corr_error, 0.000, atol=1e-3)
        assert np.isclose(
            F_O_1_ent.corrections_metadata["freysoldt_charge_correction_error"], 0.000, atol=1e-3
        )

        # test just correction returned with plot = False and return_correction_error = False:
        corr = F_O_1_ent.get_freysoldt_correction()
        assert np.isclose(corr.correction_energy, 0.11670254204631794, atol=1e-3)

        # move OUTCAR file back to original:
        shutil.move(f"{defect_path}/hidden_otcr.gz", f"{defect_path}/OUTCAR.gz")

        self._test_F_O_1_ent(
            F_O_1_ent,
            0.03146836204627482,
            "freysoldt_charge_correction",
            0.11670254204631794,
        )
        # now using Kumagai-Oba (eFNV) correction
        defect_path = f"{self.YTOS_EXAMPLE_DIR}/F_O_1/"
        # parse with no explicitly-set-charge:
        F_O_1_ent = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk/",
            dielectric=self.ytos_dielectric,
            charge_state=1,
            parse_projected_eigen=False,  # just for fast testing, not recommended in general!
        )

        self._test_F_O_1_ent(F_O_1_ent, 0.04176, "kumagai_charge_correction", 0.12699488572686776)

        # test symmetry determination (no warning here because periodicity breaking doesn't affect F_O):
        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            warnings.filterwarnings("ignore", "dict interface")  # ignore spglib warning from v2.4.1
            relaxed_defect_name = get_defect_name_from_entry(F_O_1_ent)
            print([warn.message for warn in w])  # for debugging
            assert len(w) == 0
        assert relaxed_defect_name == "F_O_D4h_Ti1.79"
        assert get_defect_name_from_entry(F_O_1_ent, relaxed=False) == "F_O_D4h_Ti1.79"

    def _test_F_O_1_ent(self, F_O_1_ent, ediff, correction_name, correction):
        assert np.isclose(F_O_1_ent.get_ediff(), ediff, atol=1e-3)
        correction_test_dict = {correction_name: correction}
        for correction_name, correction_energy in correction_test_dict.items():
            assert np.isclose(F_O_1_ent.corrections[correction_name], correction_energy, atol=1e-3)
        # assert auto-determined interstitial site is correct
        assert np.isclose(
            F_O_1_ent.defect_supercell_site.distance_and_image_from_frac_coords([0, 0, 0])[0],
            0.0,
            atol=1e-2,
        )

        return correction_test_dict

    def test_tricky_relaxed_interstitial_corrections_kumagai(self):
        """
        Test the eFNV correction performance with tricky-to-locate relaxed
        interstitial sites.

        In this test case, we look at Te_i^+1 ground-state and metastable
        structures from Kavanagh et al. 2022 doi.org/10.1039/D2FD00043A.
        """
        with warnings.catch_warnings():
            try:
                from pydefect.analyzer.calc_results import CalcResults
                from pydefect.cli.vasp.make_efnv_correction import make_efnv_correction
            except ImportError as exc:
                raise ImportError(
                    "To use the Kumagai (eFNV) charge correction, you need to install pydefect. "
                    "You can do this by running `pip install pydefect`."
                ) from exc

        def _make_calc_results(directory) -> CalcResults:
            vasprun = get_vasprun(f"{directory}/vasprun.xml.gz")
            outcar = get_outcar(f"{directory}/OUTCAR.gz")
            return CalcResults(
                structure=vasprun.final_structure,
                energy=outcar.final_energy,
                magnetization=outcar.total_mag or 0.0,
                potentials=[-p for p in outcar.electrostatic_potential],
                electronic_conv=vasprun.converged_electronic,
                ionic_conv=vasprun.converged_ionic,
            )

        bulk_calc_results = _make_calc_results(f"{self.CdTe_BULK_DATA_DIR}")

        for name, correction_energy in [
            ("Int_Te_3_Unperturbed_1", 0.2974374231312522),
            ("Int_Te_3_1", 0.3001740745077274),
        ]:
            print("Testing", name)
            defect_calc_results = _make_calc_results(f"{self.CdTe_EXAMPLE_DIR}/{name}/vasp_ncl")
            raw_efnv = make_efnv_correction(
                +1, defect_calc_results, bulk_calc_results, self.CdTe_dielectric
            )

            Te_i_ent = defect_entry_from_paths(
                defect_path=f"{self.CdTe_EXAMPLE_DIR}/{name}/vasp_ncl",
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=9.13,
                parse_projected_eigen=False,  # just for fast testing, not recommended in general!
            )

            efnv_w_doped_site = make_efnv_correction(
                +1,
                defect_calc_results,
                bulk_calc_results,
                self.CdTe_dielectric,
                defect_coords=Te_i_ent.sc_defect_frac_coords,
            )

            assert np.isclose(raw_efnv.correction_energy, efnv_w_doped_site.correction_energy, atol=1e-3)
            assert np.isclose(raw_efnv.correction_energy, sum(Te_i_ent.corrections.values()), atol=1e-3)
            assert np.isclose(raw_efnv.correction_energy, correction_energy, atol=1e-3)

            efnv_w_fcked_site = make_efnv_correction(
                +1,
                defect_calc_results,
                bulk_calc_results,
                self.CdTe_dielectric,
                defect_coords=Te_i_ent.sc_defect_frac_coords + 0.1,  # shifting to wrong defect site
                # affects correction as expected (~0.02 eV = 7% in this case)
            )
            assert not np.isclose(efnv_w_fcked_site.correction_energy, correction_energy, atol=1e-3)
            assert np.isclose(efnv_w_fcked_site.correction_energy, correction_energy, atol=1e-1)

    def test_no_dielectric_warning(self):
        """
        Test the warning about charge corrections not being possible when no
        dielectric is provided.
        """
        defect_path = f"{self.YTOS_EXAMPLE_DIR}/Int_F_-1/"

        # parse with no explicitly-set-charge:
        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk/",
                parse_projected_eigen=False,  # just for fast testing, not recommended in general!
            )
        assert len([warning for warning in w if issubclass(warning.category, UserWarning)]) == 1
        assert all(
            i in str(w[-1].message)
            for i in [
                "The dielectric constant (`dielectric`) is needed to compute finite-size charge correct",
                "Formation energies and transition levels of charged defects will likely be very inacc",
            ]
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            defect_entry_from_paths(
                defect_path=f"{self.CdTe_EXAMPLE_DIR}/v_Cd_0/vasp_ncl",
                bulk_path=f"{self.CdTe_BULK_DATA_DIR}",
            )
        assert not [warning for warning in w if issubclass(warning.category, UserWarning)]

    def test_defect_name_from_structures(self):
        # by proxy also tests defect_from_structures
        for defect_gen_name in [
            "CdTe_defect_gen",
            "ytos_defect_gen",
            "lmno_defect_gen",
            "zns_defect_gen",
        ]:
            print(f"Testing defect names for: {defect_gen_name}")
            if defect_gen_name == "zns_defect_gen":
                defect_gen = DefectsGenerator(self.non_diagonal_ZnS)
            else:
                defect_gen = DefectsGenerator.from_json(f"{self.data_dir}/{defect_gen_name}.json")

            for defect_entry in [entry for entry in defect_gen.values() if entry.charge_state == 0]:
                print(defect_entry.defect, defect_entry.defect_supercell_site)
                assert defect_name_from_structures(
                    defect_entry.bulk_supercell, defect_entry.defect_supercell
                ) == get_defect_name_from_defect(defect_entry.defect)

                # Can't use defect.structure/defect.defect_structure because might be vacancy in a 1/2
                # atom cell etc.:
                # assert defect_name_from_structures(
                #     defect_entry.defect.structure, defect_entry.defect.defect_structure
                # ) == get_defect_name_from_defect(defect_entry.defect)

    def test_defect_from_structures_rattled(self):
        """
        Test the robustness of the defect_from_structures function using
        rattled structures (note that this function is already extensively
        tested indirectly through the defect parsing tests).
        """
        from shakenbreak.distortions import rattle

        zns_defect_thermo = loadfn(f"{self.ZnS_DATA_DIR}/ZnS_thermo.json")
        v_Zn_0 = zns_defect_thermo.defect_entries["vac_1_Zn_0"]
        Al_Zn_m1 = zns_defect_thermo.defect_entries["sub_1_Al_on_Zn_-1"]
        Al_i_2 = zns_defect_thermo.defect_entries["inter_26_Al_2"]
        print(Al_i_2.defect_supercell_site.frac_coords)

        # test increasing stdev still gets correct site IDs:
        for defect_entry, type in [
            (v_Zn_0, "vacancy"),
            (Al_Zn_m1, "substitution"),
            (Al_i_2, "interstitial"),
        ]:
            for stdev in np.linspace(0.1, 1, 10):
                print(f"{defect_entry.name}, stdev: {stdev}, rattling defect supercell")
                rattled_defect_supercell = rattle(defect_entry.defect_supercell, stdev=stdev).copy()
                with warnings.catch_warnings(record=True) as w:
                    (
                        defect,
                        defect_site,  # _relaxed_ defect site in supercell (if substitution/interstitial)
                        defect_site_in_bulk,  # bulk site for vacancies/substitutions, relaxed defect site
                        # w/interstitials
                        defect_site_index,
                        bulk_site_index,
                        guessed_initial_defect_structure,
                        unrelaxed_defect_structure,
                    ) = defect_from_structures(
                        defect_entry.bulk_supercell,
                        rattled_defect_supercell,
                        return_all_info=True,
                        oxi_state="Undetermined",  # doesn't matter here so skip
                    )
                print([str(warn.message) for warn in w])  # for debugging
                if stdev >= 0.5:
                    assert (
                        "Detected atoms far from the defect site (>6.62 Å) with major displacements ("
                        ">0.5 Å) in the defect supercell. This likely indicates a mismatch"
                    ) in str(w[-1].message)
                rattled_relaxed_defect_coords = (
                    rattled_defect_supercell[
                        defect_entry.calculation_metadata["defect_site_index"]
                    ].frac_coords
                    if type != "vacancy"
                    else None
                )
                if type != "interstitial":
                    assert np.allclose(
                        defect_site_in_bulk.frac_coords,
                        defect_entry.calculation_metadata["bulk_site"].frac_coords,
                        atol=1e-2,
                    )
                    assert np.allclose(defect.site.frac_coords, [0, 0, 0], atol=1e-2)
                    assert (
                        unrelaxed_defect_structure
                        == defect_entry.calculation_metadata["unrelaxed_defect_structure"]
                    )
                else:  # interstitial
                    assert np.allclose(  # guessed initial site (closest Voronoi node)
                        guessed_initial_defect_structure[defect_site_index].frac_coords,
                        [0.53125, 0.65625, 0.125],
                        atol=stdev / 4,
                    )
                    assert np.allclose(
                        defect_site.frac_coords,
                        rattled_relaxed_defect_coords,
                        atol=1e-2,
                    )

                if type == "vacancy":
                    assert np.allclose(
                        defect_site_in_bulk.frac_coords,
                        defect_entry.defect_supercell_site.frac_coords,
                        atol=1e-2,
                    )
                    assert np.allclose(
                        defect_site.frac_coords, defect_entry.defect_supercell_site.frac_coords, atol=1e-2
                    )
                else:  # substitution/interstitial
                    assert np.allclose(
                        defect_site.frac_coords,
                        rattled_relaxed_defect_coords,
                        atol=1e-2,
                    )

                assert defect_site_index == defect_entry.calculation_metadata["defect_site_index"]
                assert bulk_site_index == defect_entry.calculation_metadata["bulk_site_index"]
                if stdev < 0.31 or type != "interstitial":  # otherwise nearest Voronoi node can differ!
                    assert (
                        guessed_initial_defect_structure
                        == defect_entry.calculation_metadata["guessed_initial_defect_structure"]
                    )

                print(f"{defect_entry.name}, stdev: {stdev}, rattling bulk supercell")  # now rattle bulk:
                with warnings.catch_warnings(record=True) as w:
                    (
                        defect,
                        defect_site,  # _relaxed_ defect site in supercell (if substitution/interstitial)
                        defect_site_in_bulk,  # bulk site for vacancies/substitutions, relaxed defect site
                        # w/interstitials
                        defect_site_index,
                        bulk_site_index,
                        guessed_initial_defect_structure,
                        unrelaxed_defect_structure,
                    ) = defect_from_structures(
                        rattle(defect_entry.bulk_supercell, stdev=stdev).copy(),
                        defect_entry.defect_supercell,
                        return_all_info=True,
                        oxi_state="Undetermined",  # doesn't matter here so skip
                    )
                print([str(warn.message) for warn in w])  # for debugging
                if stdev >= 0.5:
                    assert (
                        "Detected atoms far from the defect site (>6.62 Å) with major displacements ("
                        ">0.5 Å) in the defect supercell. This likely indicates a mismatch"
                    ) in str(w[-1].message)
                assert np.allclose(
                    defect_site_in_bulk.frac_coords,
                    defect_entry.defect_supercell_site.frac_coords,
                    atol=stdev * 3,
                )
                assert np.allclose(
                    defect_site.frac_coords, defect_entry.defect_supercell_site.frac_coords, atol=stdev * 3
                )
                assert np.allclose(
                    defect.site.frac_coords, defect_entry.defect_supercell_site.frac_coords, atol=stdev * 3
                )
                assert defect_site_index == defect_entry.calculation_metadata["defect_site_index"]
                assert bulk_site_index == defect_entry.calculation_metadata["bulk_site_index"]

    def test_point_symmetry_periodicity_breaking(self):
        """
        Test the periodicity-breaking warning with the ``point_symmetry``
        function from ``doped.utils.symmetry``.

        Note that this warning & symmetry handling is mostly tested through
        the ``DefectThermodynamics.get_symmetries_and_degeneracies()`` tests
        in ``test_thermodynamics.py``.
        """
        dp, w = _create_dp_and_capture_warnings(self.ZnS_DATA_DIR, dielectric=8.9)
        assert len(dp.defect_dict) == 17

        with warnings.catch_warnings(record=True) as w:
            point_symm, periodicity_breaking = point_symmetry_from_structure(
                dp.defect_dict["vac_1_Zn_0"].defect_supercell,
                bulk_structure=dp.defect_dict["vac_1_Zn_0"].bulk_supercell,
                return_periodicity_breaking=True,
            )
        print([str(warning.message) for warning in w])  # for debugging
        assert len(w) == 1
        assert (
            str(w[0].message)
            == "`relaxed` is set to True (i.e. get _relaxed_ defect symmetry), but doped has detected "
            "that the defect supercell is likely a non-scalar matrix expansion which could be "
            "breaking the cell periodicity and possibly preventing the correct _relaxed_ point group "
            "symmetry from being automatically determined. You can set relaxed=False to instead get "
            "the (unrelaxed) bulk site symmetry, and/or manually check/set/edit the point symmetries "
            "and corresponding orientational degeneracy factors by inspecting/editing the "
            "calculation_metadata['relaxed point symmetry']/['bulk site symmetry'] and "
            "degeneracy_factors['orientational degeneracy'] attributes."
        )
        assert periodicity_breaking
        assert point_symm == "C1"

        for name, defect_entry in dp.defect_dict.items():
            print(f"Checking symmetry for {name}")
            with warnings.catch_warnings(record=True) as w:
                assert point_symmetry_from_structure(defect_entry.defect_supercell) == "C1"
            print([str(warning.message) for warning in w])  # for debugging
            assert not w  # no warnings with just defect supercell as can't determine periodicity breaking
            with warnings.catch_warnings(record=True) as w:
                assert point_symmetry_from_structure(
                    defect_entry.defect_supercell, defect_entry.bulk_supercell, relaxed=False
                ) in ["Td", "C3v", "Cs", "C1"]
            print([str(warning.message) for warning in w])  # for debugging
            assert not w  # no periodicity breaking warning with `relaxed=False`
            with pytest.raises(RuntimeError) as excinfo:
                point_symmetry_from_structure(defect_entry.defect_supercell, relaxed=False)
            assert "Please also supply the unrelaxed bulk structure" in str(excinfo.value)

    def test_bulk_defect_compatibility_checks(self):
        """
        Test our bulk/defect INCAR/POSCAR/KPOINTS/POTCAR compatibility checks.

        Note that _compatible_ cases are indirectly tested above, by checking
        the number of warnings is as expected (i.e. no compatibility warnings
        when matching defect/bulk settings used).
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            defect_entry_from_paths(
                defect_path=f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_Unperturbed_1/vasp_ncl",
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=9.13,
                skip_corrections=True,
                parse_projected_eigen=False,  # just for fast testing, not recommended in general!
            )
            print([str(warning.message) for warning in w])  # for debugging
            assert len(w) == 1
            assert all(
                i in str(w[-1].message)
                for i in [
                    "There are mismatching INCAR tags for your defect and bulk calculations",
                    "[('ADDGRID', False, True)]",
                ]
            )

        # edit vasprun.xml.gz to have different INCAR tags:
        with (
            gzip.open(
                f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_Unperturbed_1/vasp_ncl/vasprun.xml.gz", "rb"
            ) as f_in,
            open("./vasprun.xml", "wb") as f_out,
        ):
            shutil.copyfileobj(f_in, f_out)

        # open vasprun.xml, edit ENCUT and add LDAU to INCAR but with default value:
        with open("./vasprun.xml") as f:  # line 11 (10 in python indexing) is start of INCAR
            lines = f.readlines()
            for i, line in enumerate(lines):
                if '<i name="ENCUT">' in line:
                    lines[i] = lines[i].replace("450", "500")
                    break

            new_vr_lines = lines[:11] + ['  <i type="logical" name="LDAU"> F  </i>\n'] + lines[11:]

        with open("./vasprun.xml", "w") as f_out:
            f_out.writelines(new_vr_lines)

        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            defect_entry_from_paths(
                defect_path=".",
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=9.13,
                skip_corrections=True,
                parse_projected_eigen=False,  # just for fast testing, not recommended in general!
            )
            print([str(warning.message) for warning in w])  # for debugging
            assert len(w) == 1
            assert all(
                i in str(w[-1].message)
                for i in [
                    "There are mismatching INCAR tags for your defect and bulk calculations",
                    "[('ADDGRID', False, True), ('ENCUT', 500.0, 450.0)]",
                ]
            )

        # edit KPOINTS:
        for i, line in enumerate(lines):  # vasprun lines already loaded above
            if "   <v>       0.50000000       0.50000000       0.50000000 </v>" in line:
                lines[i] = lines[i].replace("0.500000", "0.125")
                break

        with open("./vasprun.xml", "w") as f_out:
            f_out.writelines(lines)

        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            defect_entry_from_paths(
                defect_path=".",
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=9.13,
                skip_corrections=True,
                parse_projected_eigen=False,  # just for fast testing, not recommended in general!
            )
            print([str(warning.message) for warning in w])  # for debugging
            assert len(w) == 2  # now INCAR and KPOINTS warnings!
            assert any(
                all(
                    i in str(warning.message)
                    for i in [
                        "The KPOINTS for your defect and bulk calculations do not match",
                        "[0.125, 0.125, 0.125]",
                    ]
                )
                for warning in w
            )

        # edit POTCAR symbols:
        for i, line in enumerate(lines):  # vasprun lines already loaded above
            if "PAW_PBE Cd 06Sep2000" in line:
                lines[i] = lines[i].replace("Cd", "Cd_GW")
                break

        with open("./vasprun.xml", "w") as f_out:
            f_out.writelines(lines)

        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            defect_entry_from_paths(
                defect_path=".",
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=9.13,
                skip_corrections=True,
                charge_state=+1,  # manually specify charge state here, as our edited POTCAR doesn't exist
                parse_projected_eigen=False,  # just for fast testing, not recommended in general!
            )
            print([str(warning.message) for warning in w])  # for debugging
            assert len(w) == 3  # now INCAR and KPOINTS and POTCAR warnings!
            assert any(
                all(
                    i in str(warning.message)
                    for i in [
                        "The POTCAR symbols for your defect and bulk calculations do not match",
                        "PAW_PBE Cd 06Sep2000",
                        "PAW_PBE Cd_GW 06Sep2000",
                    ]
                )
                for warning in w
            )

        # edit POSCAR volume:
        for i, line in enumerate(lines[-1000:]):  # vasprun lines already loaded above
            if "13.08676800" in line:
                lines[i + len(lines) - 1000] = line.replace("13.08676800", "3000")  # went to the year 3000
                break

        with open("./vasprun.xml", "w") as f_out:
            f_out.writelines(lines)

        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            defect_entry_from_paths(
                defect_path=".",
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=9.13,
                skip_corrections=True,
                charge_state=+1,  # manually specify charge state here, as our edited POTCAR doesn't exist
                parse_projected_eigen=False,  # just for fast testing, not recommended in general!
            )
            print([str(warning.message) for warning in w])  # for debugging
            assert any(
                "The defect and bulk supercells are not the same size, having volumes of 513790.5 and "
                "2241.3 Å^3 respectively." in str(warning.message)
                for warning in w
            )

    def test_checking_defect_bulk_cell_definitions(self):
        with warnings.catch_warnings(record=True) as w:
            DefectParser.from_paths(
                defect_path=f"{self.data_dir}/Doped_CdTe",
                bulk_path=self.CdTe_BULK_DATA_DIR,
                skip_corrections=True,
                parse_projected_eigen=False,  # just for fast testing, not recommended in general!
            )
        assert any("Detected atoms far from the defect site" in str(warning.message) for warning in w)

    @custom_mpl_image_compare(filename="Cu2SiSe3_v_Cu_0_eigenvalue_plot.png")
    def test_eigenvalues_parsing_and_warnings(self):
        """
        Test eigenvalues functions.

        Print statements added because dict comparison doesn't give verbose
        output on exact location of failure.
        """

        def _compare_band_edge_states_dicts(d1, d2, orb_diff_tol: float = 0.1):
            """
            Compare two dictionaries of band edge states, removing metadata
            keys and allowing a slight difference in the
            ``vbm/cbm_orbital_diff`` values to account for rounding errors with
            ``PROCAR``s.
            """
            if isinstance(d1, str):
                d1 = loadfn(d1)
            if isinstance(d2, str):
                d2 = loadfn(d2)

            d1 = d1.as_dict()
            d2 = d2.as_dict()

            cbm_orbital_diffs1 = [subdict.pop("cbm_orbital_diff") for subdict in d1["states"]]
            cbm_orbital_diffs2 = [subdict.pop("cbm_orbital_diff") for subdict in d2["states"]]
            for i, j in zip(cbm_orbital_diffs1, cbm_orbital_diffs2, strict=False):
                print(f"cbm_orbital_diffs: {i:.3f} vs {j:.3f}")
                assert np.isclose(i, j, atol=orb_diff_tol)
            vbm_orbital_diffs1 = [subdict.pop("vbm_orbital_diff") for subdict in d1["states"]]
            vbm_orbital_diffs2 = [subdict.pop("vbm_orbital_diff") for subdict in d2["states"]]
            for i, j in zip(vbm_orbital_diffs1, vbm_orbital_diffs2, strict=False):
                print(f"vbm_orbital_diffs: {i:.3f} vs {j:.3f}")
                assert np.isclose(i, j, atol=orb_diff_tol)

            orb_infos_orbitals1 = [
                subdict["vbm_info"]["orbital_info"].pop("orbitals") for subdict in d1["states"]
            ] + [subdict["cbm_info"]["orbital_info"].pop("orbitals") for subdict in d1["states"]]
            orb_infos_orbitals2 = [
                subdict["vbm_info"]["orbital_info"].pop("orbitals") for subdict in d2["states"]
            ] + [subdict["cbm_info"]["orbital_info"].pop("orbitals") for subdict in d2["states"]]
            for i, j in zip(orb_infos_orbitals1, orb_infos_orbitals2, strict=False):
                print(f"orbital_info_orbitals: {i} vs {j}")
                for k, v in i.items():
                    assert np.allclose(v, j[k], atol=orb_diff_tol)

            participation_ratio1 = (
                [
                    subdict["vbm_info"]["orbital_info"].pop("participation_ratio")
                    for subdict in d1["states"]
                ]
                + [
                    subdict["cbm_info"]["orbital_info"].pop("participation_ratio")
                    for subdict in d1["states"]
                ]
                + [
                    subsubdict.pop("participation_ratio")
                    for subdict in d1["states"]
                    for subsubdict in subdict["localized_orbitals"]
                ]
            )

            participation_ratio2 = (
                [
                    subdict["vbm_info"]["orbital_info"].pop("participation_ratio")
                    for subdict in d2["states"]
                ]
                + [
                    subdict["cbm_info"]["orbital_info"].pop("participation_ratio")
                    for subdict in d2["states"]
                ]
                + [
                    subsubdict.pop("participation_ratio")
                    for subdict in d2["states"]
                    for subsubdict in subdict["localized_orbitals"]
                ]
            )
            print(f"participation_ratio: {participation_ratio1} vs {participation_ratio2}")
            assert np.allclose(participation_ratio1, participation_ratio2, atol=orb_diff_tol)

            assert _remove_metadata_keys_from_dict(d1) == _remove_metadata_keys_from_dict(d2)

        # Test loading of MgO using vasprun.xml
        defect_entry = DefectParser.from_paths(
            f"{self.MgO_EXAMPLE_DIR}/Defects/Pre_Calculated_Results/Mg_O_+1/vasp_std",
            f"{self.MgO_EXAMPLE_DIR}/Defects/Pre_Calculated_Results/MgO_bulk/vasp_std",
            skip_corrections=True,
            parse_projected_eigen=True,
        ).defect_entry
        assert defect_entry.degeneracy_factors["spin degeneracy"] == 2

        print("Testing MgO eigenvalue analysis")
        bes, fig = defect_entry.get_eigenvalue_analysis()  # Test plotting KS
        Mg_O_1_bes_path = (
            f"{self.MgO_EXAMPLE_DIR}/Defects/Pre_Calculated_Results/Mg_O_1_band_edge_states.json"
        )
        # dumpfn(bes, Mg_O_1_bes_path)  # for saving test data
        _compare_band_edge_states_dicts(bes, Mg_O_1_bes_path, orb_diff_tol=0.01)
        assert bes.has_occupied_localized_state
        assert not any(
            [bes.has_acceptor_phs, bes.has_donor_phs, bes.has_unoccupied_localized_state, bes.is_shallow]
        )

        # Test loading using ``DefectsParser``
        print("Testing Cu2SiSe3 eigenvalue analysis default, with vaspruns (& PROCARs)")
        dp = DefectsParser(f"{self.Cu2SiSe3_EXAMPLE_DIR}", skip_corrections=True)

        print("Testing v_Cu_0 with plot = True")
        bes, fig = dp.defect_dict["v_Cu_0"].get_eigenvalue_analysis()  # Test plotting KS
        v_Cu_0_bes_path = f"{self.Cu2SiSe3_EXAMPLE_DIR}/Cu2SiSe3_vac_band_edge_states.json"
        # dumpfn(bes, v_Cu_0_bes_path)  # for saving test data
        _compare_band_edge_states_dicts(bes, v_Cu_0_bes_path, orb_diff_tol=0.001)
        assert bes.has_acceptor_phs
        assert bes.is_shallow
        assert not any(
            [bes.has_donor_phs, bes.has_occupied_localized_state, bes.has_unoccupied_localized_state]
        )
        assert dp.defect_dict["v_Cu_0"].is_shallow

        print("Testing v_Cu_0 with plot = False")
        bes2 = dp.defect_dict["v_Cu_0"].get_eigenvalue_analysis(
            plot=False,
        )  # Test getting BES and not plot
        _compare_band_edge_states_dicts(bes, bes2, orb_diff_tol=0.001)
        assert bes.has_acceptor_phs
        assert bes.is_shallow
        assert not any(
            [bes.has_donor_phs, bes.has_occupied_localized_state, bes.has_unoccupied_localized_state]
        )

        print("Testing Si_i_-1 with plot = True")
        bes = dp.defect_dict["Si_i_-1"].get_eigenvalue_analysis(plot=False)
        Si_i_m1_bes_path = f"{self.Cu2SiSe3_EXAMPLE_DIR}/Cu2SiSe3_int_band_edge_states.json"
        # dumpfn(bes, Si_i_m1_bes_path)  # for saving test data
        _compare_band_edge_states_dicts(bes, Si_i_m1_bes_path, orb_diff_tol=0.01)
        assert bes.has_occupied_localized_state
        assert not any(
            [bes.has_acceptor_phs, bes.has_donor_phs, bes.has_unoccupied_localized_state, bes.is_shallow]
        )

        # test parses fine without projected eigenvalues from vaspruns:
        print("Testing Cu2SiSe3 parsing and eigenvalue analysis with bulk vr, no projected eigenvalues")
        # first without bulk:
        shutil.move(
            f"{self.Cu2SiSe3_EXAMPLE_DIR}/bulk/vasp_std/vasprun.xml.gz",
            f"{self.Cu2SiSe3_EXAMPLE_DIR}/bulk/vasp_std/hidden_vr.gz",
        )
        shutil.copy(
            f"{self.Cu2SiSe3_EXAMPLE_DIR}/bulk/vasp_std/no_eig_vr.xml.gz",
            f"{self.Cu2SiSe3_EXAMPLE_DIR}/bulk/vasp_std/vasprun.xml.gz",
        )
        dp = DefectsParser(f"{self.Cu2SiSe3_EXAMPLE_DIR}", skip_corrections=True)

        print("Testing v_Cu_0 with plot = True")
        bes, fig = dp.defect_dict["v_Cu_0"].get_eigenvalue_analysis()  # Test plotting KS
        _compare_band_edge_states_dicts(bes, v_Cu_0_bes_path, orb_diff_tol=0.1)

        print("Testing Si_i_-1 with plot = True")
        bes = dp.defect_dict["Si_i_-1"].get_eigenvalue_analysis(plot=False)
        _compare_band_edge_states_dicts(bes, Si_i_m1_bes_path, orb_diff_tol=0.1)

        # then without defect vasprun (in one case):
        print(
            "Testing Cu2SiSe3 parsing and eigenvalue analysis without v_Cu_0 vasprun with projected "
            "eigenvalues, or bulk vasprun with projected eigenvalues (but with PROCARs)"
        )
        shutil.move(
            f"{self.Cu2SiSe3_EXAMPLE_DIR}/v_Cu_0/vasp_std/vasprun.xml.gz",
            f"{self.Cu2SiSe3_EXAMPLE_DIR}/v_Cu_0/vasp_std/hidden_vr.gz",
        )
        shutil.copy(
            f"{self.Cu2SiSe3_EXAMPLE_DIR}/v_Cu_0/vasp_std/no_eig_vr.xml.gz",
            f"{self.Cu2SiSe3_EXAMPLE_DIR}/v_Cu_0/vasp_std/vasprun.xml.gz",
        )
        dp = DefectsParser(f"{self.Cu2SiSe3_EXAMPLE_DIR}", skip_corrections=True)

        print("Testing v_Cu_0 with plot = True")
        bes, fig = dp.defect_dict["v_Cu_0"].get_eigenvalue_analysis()  # Test plotting KS
        _compare_band_edge_states_dicts(bes, v_Cu_0_bes_path, orb_diff_tol=0.1)

        print("Testing Si_i_-1 with plot = True")
        bes = dp.defect_dict["Si_i_-1"].get_eigenvalue_analysis(plot=False)
        _compare_band_edge_states_dicts(bes, Si_i_m1_bes_path, orb_diff_tol=0.1)

        # then without defect vasprun w/eig but with bulk vasprun w/eig:
        print(
            "Testing Cu2SiSe3 parsing and eigenvalue analysis without v_Cu_0 vasprun with projected "
            "eigenvalues, but with bulk vasprun with eigenvalues"
        )
        shutil.move(
            f"{self.Cu2SiSe3_EXAMPLE_DIR}/bulk/vasp_std/hidden_vr.gz",
            f"{self.Cu2SiSe3_EXAMPLE_DIR}/bulk/vasp_std/vasprun.xml.gz",
        )
        dp = DefectsParser(f"{self.Cu2SiSe3_EXAMPLE_DIR}", skip_corrections=True)

        print("Testing v_Cu_0 with plot = True")
        bes, fig = dp.defect_dict["v_Cu_0"].get_eigenvalue_analysis()  # Test plotting KS
        _compare_band_edge_states_dicts(bes, v_Cu_0_bes_path, orb_diff_tol=0.1)

        print("Testing Si_i_-1 with plot = True")
        bes = dp.defect_dict["Si_i_-1"].get_eigenvalue_analysis(plot=False)
        _compare_band_edge_states_dicts(bes, Si_i_m1_bes_path, orb_diff_tol=0.1)

        # Test non-collinear calculation
        print("Testing CdTe parsing and eigenvalue analysis; SOC example case")
        defect_entry = DefectParser.from_paths(
            f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_1/vasp_ncl",
            f"{self.CdTe_EXAMPLE_DIR}/CdTe_bulk/vasp_ncl",
            skip_corrections=True,
            parse_projected_eigen=True,
        ).defect_entry

        print("Testing Int_Te_3_1 with plot = True")
        bes = defect_entry.get_eigenvalue_analysis(plot=False)
        Te_i_1_SOC_bes_path = f"{self.CdTe_EXAMPLE_DIR}/CdTe_test_soc_band_edge_states.json"
        # dumpfn(bes, Te_i_1_SOC_bes_path)  # for saving test data
        _compare_band_edge_states_dicts(bes, Te_i_1_SOC_bes_path, orb_diff_tol=0.02)
        assert bes.has_unoccupied_localized_state
        assert not any(
            [bes.has_acceptor_phs, bes.has_donor_phs, bes.has_occupied_localized_state, bes.is_shallow]
        )

        # Test warning for no projected orbitals: Sb2Se3 data
        print("Testing Sb2Se3 parsing and warning; no projected orbital data")
        with warnings.catch_warnings(record=True) as w:
            DefectParser.from_paths(
                f"{self.Sb2Se3_DATA_DIR}/defect/O_1",
                f"{self.Sb2Se3_DATA_DIR}/bulk",
                skip_corrections=True,
                parse_projected_eigen=True,
            )

        print([str(warning.message) for warning in w])  # for debugging
        assert any("Could not parse eigenvalue data" in str(warning.message) for warning in w)

        # Test no warning for no projected orbitals with default ``parse_projected_eigen=None`` (attempt
        # but don't warn): Sb2Se3 data
        print("Testing Sb2Se3 parsing and warning; no warning with default settings")
        with warnings.catch_warnings(record=True) as w:
            DefectParser.from_paths(
                f"{self.Sb2Se3_DATA_DIR}/defect/O_1",
                f"{self.Sb2Se3_DATA_DIR}/bulk",
                skip_corrections=True,
            )

        print([str(warning.message) for warning in w])  # for debugging
        assert not w

        # test parsing fine when eigenvalue data not originally parsed, but then
        # ``DefectEntry.get_eigenvalue_analysis()`` later called:
        # Test loading using ``PROCAR`` and ``DefectsParser``
        print("Testing Cu2SiSe3 eigenvalue analysis, without parsing eigenvalue data originally")
        shutil.move(
            f"{self.Cu2SiSe3_EXAMPLE_DIR}/v_Cu_0/vasp_std/hidden_vr.gz",
            f"{self.Cu2SiSe3_EXAMPLE_DIR}/v_Cu_0/vasp_std/vasprun.xml.gz",
        )
        dp = DefectsParser(
            f"{self.Cu2SiSe3_EXAMPLE_DIR}", skip_corrections=True, parse_projected_eigen=False
        )

        # now should still all work fine:
        print("Testing v_Cu_0 with plot = True")
        bes, fig = dp.defect_dict["v_Cu_0"].get_eigenvalue_analysis()  # Test plotting KS
        _compare_band_edge_states_dicts(bes, v_Cu_0_bes_path, orb_diff_tol=0.001)

        print("Testing Si_i_-1 with plot = True")
        bes = dp.defect_dict["Si_i_-1"].get_eigenvalue_analysis(plot=False)
        _compare_band_edge_states_dicts(bes, Si_i_m1_bes_path, orb_diff_tol=0.001)

        # test directly using ``get_eigenvalue_analysis`` with VASP outputs:
        # now should still all work fine:
        print("Testing v_Cu_0 with plot = True, direct VASP outputs; vaspruns")
        bes, fig = get_eigenvalue_analysis(
            bulk_vr=get_vasprun(
                f"{self.Cu2SiSe3_EXAMPLE_DIR}/bulk/vasp_std/vasprun.xml.gz", parse_projected_eigen=True
            ),
            defect_vr=get_vasprun(
                f"{self.Cu2SiSe3_EXAMPLE_DIR}/v_Cu_0/vasp_std/vasprun.xml.gz", parse_projected_eigen=True
            ),
        )
        _compare_band_edge_states_dicts(bes, v_Cu_0_bes_path, orb_diff_tol=0.001)

        print("Testing v_Cu_0 with plot = True, direct VASP outputs; vaspruns and procars")
        self.tearDown()  # ensure PROCARs returned to original state
        bes, fig = get_eigenvalue_analysis(
            bulk_vr=get_vasprun(
                f"{self.Cu2SiSe3_EXAMPLE_DIR}/bulk/vasp_std/vasprun.xml.gz", parse_projected_eigen=True
            ),
            defect_vr=get_vasprun(
                f"{self.Cu2SiSe3_EXAMPLE_DIR}/v_Cu_0/vasp_std/vasprun.xml.gz", parse_projected_eigen=True
            ),
            bulk_procar=get_procar(f"{self.Cu2SiSe3_EXAMPLE_DIR}/bulk/vasp_std/PROCAR.gz"),
            defect_procar=get_procar(f"{self.Cu2SiSe3_EXAMPLE_DIR}/v_Cu_0/vasp_std/PROCAR.gz"),
        )
        _compare_band_edge_states_dicts(bes, v_Cu_0_bes_path, orb_diff_tol=0.1)

        # test error when not providing defect_entry or bulk_vr:
        with pytest.raises(ValueError) as exc:
            get_eigenvalue_analysis()
        assert (
            "If `defect_entry` is not provided, then both `bulk_vr` and `defect_vr` at a minimum must be "
            "provided!" in str(exc.value)
        )

        # test all fine when saving and reloading from JSON (previously didn't, but fixed)
        dumpfn(dp.defect_dict, "test.json")
        reloaded_defect_dict = loadfn("test.json")
        bes, fig = reloaded_defect_dict["v_Cu_0"].get_eigenvalue_analysis()
        _compare_band_edge_states_dicts(bes, v_Cu_0_bes_path, orb_diff_tol=0.001)
        os.remove("test.json")

        return fig

    @custom_mpl_image_compare("YTOS_Int_F_-1_eigenvalue_plot_ylim.png")
    def test_eigenvalue_ylim_customisation(self):
        """
        Test parsing of extrinsic F in YTOS interstitial and Kumagai-Oba (eFNV)
        correction, then outputting the eigenvalue plot with a custom ylim
        setting.
        """
        with warnings.catch_warnings(record=True) as w:
            int_F_minus1_ent = defect_entry_from_paths(
                defect_path=f"{self.YTOS_EXAMPLE_DIR}/Int_F_-1",
                bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk",
                dielectric=self.ytos_dielectric,
            )
        assert not [warning for warning in w if issubclass(warning.category, UserWarning)]
        bes, eig_fig = int_F_minus1_ent.get_eigenvalue_analysis(ylims=(-5, 5))
        assert not any(
            [
                bes.has_acceptor_phs,
                bes.has_donor_phs,
                bes.has_occupied_localized_state,
                bes.has_unoccupied_localized_state,
                bes.is_shallow,
            ]
        )

        return eig_fig

    @custom_mpl_image_compare("YTOS_Int_F_-1_eigenvalue_plot_legend.png")
    def test_eigenvalue_legend_customisation(self):
        """
        Test parsing of extrinsic F in YTOS interstitial and Kumagai-Oba (eFNV)
        correction, then outputting the eigenvalue plot with a custom ylim
        setting.
        """
        with warnings.catch_warnings(record=True) as w:
            int_F_minus1_ent = defect_entry_from_paths(
                defect_path=f"{self.YTOS_EXAMPLE_DIR}/Int_F_-1",
                bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk",
                dielectric=self.ytos_dielectric,
            )
        assert not [warning for warning in w if issubclass(warning.category, UserWarning)]
        bes, eig_fig = int_F_minus1_ent.get_eigenvalue_analysis(
            legend_kwargs={"loc": "lower left", "ncol": 2, "fontsize": 12}
        )
        assert not any(
            [
                bes.has_acceptor_phs,
                bes.has_donor_phs,
                bes.has_occupied_localized_state,
                bes.has_unoccupied_localized_state,
                bes.is_shallow,
            ]
        )

        return eig_fig

    @custom_mpl_image_compare("YTOS_Int_F_-1_eigenvalue_plot_no_legend.png")
    def test_eigenvalue_no_legend(self):
        """
        Test parsing of extrinsic F in YTOS interstitial and Kumagai-Oba (eFNV)
        correction, then outputting the eigenvalue plot with a custom ylim
        setting.
        """
        with warnings.catch_warnings(record=True) as w:
            int_F_minus1_ent = defect_entry_from_paths(
                defect_path=f"{self.YTOS_EXAMPLE_DIR}/Int_F_-1",
                bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk",
                dielectric=self.ytos_dielectric,
            )
        assert not [warning for warning in w if issubclass(warning.category, UserWarning)]
        bes, eig_fig = int_F_minus1_ent.get_eigenvalue_analysis(legend_kwargs=False)
        assert not any(
            [
                bes.has_acceptor_phs,
                bes.has_donor_phs,
                bes.has_occupied_localized_state,
                bes.has_unoccupied_localized_state,
                bes.is_shallow,
            ]
        )

        return eig_fig

    @custom_mpl_image_compare("CdTe_v_Cd_-1_eigenvalue_plot.png")
    def test_eigenvalue_criteria(self):
        """
        Test eigenvalue plotting for v_Cd_-1 in CdTe, and customising the
        orbital and energy similarity criteria.
        """
        CdTe_defect_thermo = loadfn(os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_thermo_wout_meta.json.gz"))
        v_Cd_minus1 = CdTe_defect_thermo.defect_entries["v_Cd_-1"]
        with warnings.catch_warnings(record=True) as w:
            bes, fig = v_Cd_minus1.get_eigenvalue_analysis()
        print([str(warning.message) for warning in w])  # for debugging
        assert not w
        assert bes.has_unoccupied_localized_state  # with pydefect defaults this hole polaron state isn't
        # identified as the orbital similarity to the VBM is relatively high, but updated doped defaults
        # work

        with warnings.catch_warnings(record=True) as w:
            bes, fig = v_Cd_minus1.get_eigenvalue_analysis(
                similar_orb_criterion=0.2, similar_energy_criterion=0.5
            )
            # pydefect default
        print([str(warning.message) for warning in w])  # for debugging
        assert not w
        assert not bes.has_unoccupied_localized_state  # no longer identified

        with warnings.catch_warnings(record=True) as w:
            bes, fig = v_Cd_minus1.get_eigenvalue_analysis(
                similar_orb_criterion=0.01, similar_energy_criterion=0.01
            )
            # fails so is dynamically updated with a warning
        print([str(warning.message) for warning in w])  # for debugging
        assert len(w) == 1
        assert (
            "Band-edge state identification failed with the current criteria: "
            "similar_orb_criterion=0.01, similar_energy_criterion=0.01 eV, but succeeded with "
            "similar_orb_criterion=0.35, similar_energy_criterion=0.5 eV." in str(w[0].message)
        )
        assert not bes.has_unoccupied_localized_state  # no longer identified

        return fig

    # no longer warned because actually can have orientational degeneracy < 1 for non-trivial defects
    # like split-vacancies, antisite swaps, split-interstitials etc
    # def test_orientational_degeneracy_error(self):
    #     # note most usages of get_orientational_degeneracy are tested (indirectly) via the
    #     # DefectsParser/DefectThermodynamics tests
    #     for defect_type in ["vacancy", "substitution", DefectType.Vacancy, DefectType.Substitution]:
    #         print(defect_type)  # for debugging
    #         with pytest.raises(ValueError) as exc:
    #             get_orientational_degeneracy(
    #                 relaxed_point_group="Td", bulk_site_point_group="C3v", defect_type=defect_type
    #             )
    #         assert (
    #             "From the input/determined point symmetries, an orientational degeneracy factor of 0.25 "
    #             "is predicted, which is less than 1, which is not reasonable for "
    #             "vacancies/substitutions, indicating an error in the symmetry determination!"
    #         ) in str(exc.value)
    #
    #     for defect_type in ["interstitial", DefectType.Interstitial]:
    #         print(defect_type)  # for debugging
    #         orientational_degeneracy = get_orientational_degeneracy(
    #             relaxed_point_group="Td", bulk_site_point_group="C3v", defect_type=defect_type
    #         )
    #         assert np.isclose(orientational_degeneracy, 0.25, atol=1e-2)

    def test_magnetization_parsing(self):
        # individual checks first:
        # bulk NCL:
        vr = get_vasprun(f"{self.CdTe_BULK_DATA_DIR}/vasprun.xml.gz", parse_projected_eigen=True)
        assert np.allclose(get_magnetization_from_vasprun(vr), 0, atol=0.02)
        assert spin_degeneracy_from_vasprun(vr) == 1

        # -1 ncl:
        vr = get_vasprun(
            f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-1/vasp_ncl/vasprun.xml.gz", parse_projected_eigen=True
        )
        assert np.isclose(np.linalg.norm(get_magnetization_from_vasprun(vr)), 1, atol=0.05)
        assert spin_degeneracy_from_vasprun(vr) == 2

        # S = 0 bipolaron ncl:
        vr = get_vasprun(
            f"{self.data_dir}/Magnetization_Tests/CdTe/v_Cd_C2v_Bipolaron_S0_0/vasp_ncl/vasprun.xml.gz",
            parse_projected_eigen=True,
        )
        assert np.isclose(np.linalg.norm(get_magnetization_from_vasprun(vr)), 0.903, atol=0.05)
        assert spin_degeneracy_from_vasprun(vr) == 1

        # S = 1 bipolaron ncl:
        vr = get_vasprun(
            f"{self.data_dir}/Magnetization_Tests/CdTe/v_Cd_C2v_Bipolaron_S1_0/vasp_ncl/vasprun.xml.gz",
            parse_projected_eigen=True,
        )
        assert np.isclose(np.linalg.norm(get_magnetization_from_vasprun(vr)), 1.6, atol=0.05)
        assert spin_degeneracy_from_vasprun(vr) == 3

        # O2 triplet calculation, vasp_std, ISPIN = 2
        vr = get_vasprun(
            f"{self.data_dir}/Magnetization_Tests/O2_mmm_EaH_0/vasp_std/vasprun.xml.gz",
            parse_projected_eigen=True,
        )
        print(get_magnetization_from_vasprun(vr))
        assert get_magnetization_from_vasprun(vr) == 2
        assert spin_degeneracy_from_vasprun(vr) == 3

        # O2 triplet calculation, vasp_ncl (near-perfect triplet)
        vr = get_vasprun(
            f"{self.data_dir}/Magnetization_Tests/O2_mmm_EaH_0/vasp_ncl/vasprun.xml.gz",
            parse_projected_eigen=True,
        )
        assert np.isclose(np.linalg.norm(get_magnetization_from_vasprun(vr)), 2, atol=0.001)
        assert spin_degeneracy_from_vasprun(vr) == 3

        # F_i_-1, vasp_std, ISPIN = 1 (non spin polarised)
        int_F_minus1_ent = defect_entry_from_paths(
            defect_path=f"{self.YTOS_EXAMPLE_DIR}/Int_F_-1/",
            bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk/",
            dielectric=self.ytos_dielectric,
        )
        assert int_F_minus1_ent.degeneracy_factors["spin degeneracy"] == 1
        vr = get_vasprun(f"{self.YTOS_EXAMPLE_DIR}/Int_F_-1/vasprun.xml.gz", parse_projected_eigen=True)
        assert get_magnetization_from_vasprun(vr) == 0

        # test DefectsParser handling:
        dp, w = _create_dp_and_capture_warnings(
            output_path=f"{self.data_dir}/Magnetization_Tests/CdTe",
            bulk_path=f"{self.CdTe_BULK_DATA_DIR}",
            dielectric=9.13,
        )
        assert dp.defect_dict["v_Cd_C2v_Bipolaron_S0_0"].degeneracy_factors["spin degeneracy"] == 1
        assert dp.defect_dict["v_Cd_C2v_Bipolaron_S1_0"].degeneracy_factors["spin degeneracy"] == 3
        self._check_DefectsParser(dp)

    def test_bulk_symprec_and_periodicity_breaking_checks(self):
        """
        ``Int_F_-1`` in YTOS is a good test case here.

        The determined bulk site
        symmetry of this defect is sensitive to the choice of ``symprec``, and
        ``spglib`` sometimes gives both ``C1`` and ``Cs`` site symmetries for
        the generated equivalent sites with default bulk ``symprec=0.01``
        (which previously caused false periodicity breaking warnings).
        """
        symprec_settings_and_expected_syms = [
            ({"bulk_symprec": 0.01}, "C4v", "C4v"),
            ({"bulk_symprec": 0.008}, "Cs", "C4v"),
            ({"bulk_symprec": 0.005}, "Cs", "C4v"),
            ({"bulk_symprec": 0.0025}, "C1", "C4v"),
            ({"symprec": 0.01}, "C4v", "Cs"),
            ({"symprec": 0.1}, "C4v", "C4v"),
            ({}, "C4v", "C4v"),
        ]
        for symprec_settings, expected_site_sym, expected_relax_sym in [
            ({}, "C4v", "C4v"),
            *symprec_settings_and_expected_syms,
        ]:
            print(f"Testing with {symprec_settings} and expected site symmetry {expected_site_sym}")
            dp, w = _create_dp_and_capture_warnings(
                output_path=self.YTOS_EXAMPLE_DIR, dielectric=self.ytos_dielectric, **symprec_settings
            )
            assert not w
            self._check_DefectsParser(dp)
            with warnings.catch_warnings(record=True) as w:
                sym_degen_df = dp.get_defect_thermodynamics().get_symmetries_and_degeneracies()
            print([str(warning.message) for warning in w])
            assert not w

            expected_site_multiplicity = {"C4v": 2.0, "Cs": 8.0, "C1": 16.0}[expected_site_sym]
            with warnings.catch_warnings(record=True) as w:
                assert (
                    point_symmetry_from_defect_entry(
                        dp.defect_dict["Int_F_-1"], relaxed=True, symprec=symprec_settings.get("symprec")
                    )
                    == expected_relax_sym
                )
                assert (
                    point_symmetry_from_defect_entry(
                        dp.defect_dict["Int_F_-1"],
                        relaxed=False,
                        symprec=symprec_settings.get("bulk_symprec"),
                    )
                    == expected_site_sym
                )
                assert (
                    dp.defect_dict["Int_F_-1"].defect.get_multiplicity(
                        symprec=symprec_settings.get("bulk_symprec")
                    )
                    == expected_site_multiplicity
                )

            # for bulk_symprec = 0.01, we have a borderline situation where this fixed symprec value
            # gives both C1 and Cs symmetries for the returned equivalent sites, but with a multiplicity
            # only matching C4v; slight adjustments to symprec (as done automatically when bulk_symprec is
            # not set) or dist_tol avoid this
            expected_orientational_degeneracy = get_orientational_degeneracy(
                relaxed_point_group=expected_relax_sym,
                bulk_site_point_group=expected_site_sym,
                **symprec_settings,
            )
            print(sym_degen_df.loc[("Int_F", "-1")])
            assert list(sym_degen_df.loc[("Int_F", "-1")]) == [
                expected_site_sym,
                expected_relax_sym,
                expected_orientational_degeneracy,
                1,
                expected_orientational_degeneracy,
                expected_site_multiplicity,
            ]

            for regen_symprec_settings, regen_expected_site_sym, regen_expected_relax_sym in [
                ({}, expected_site_sym, expected_relax_sym),  # test stays the same if settings not changed
                *symprec_settings_and_expected_syms,
            ]:
                print(
                    f"Testing sym_degen_df re-generation with {regen_symprec_settings} and expected site "
                    f"symmetry {expected_site_sym}"
                )
                with warnings.catch_warnings(record=True) as w:
                    sym_degen_df = dp.get_defect_thermodynamics().get_symmetries_and_degeneracies(
                        **regen_symprec_settings
                    )
                print([str(warning.message) for warning in w])
                assert not w
                expected_site_multiplicity = {"C4v": 2.0, "Cs": 8.0, "C1": 16.0}[regen_expected_site_sym]
                expected_orientational_degeneracy = get_orientational_degeneracy(
                    relaxed_point_group=regen_expected_relax_sym,
                    bulk_site_point_group=regen_expected_site_sym,
                    **regen_symprec_settings,
                )
                print(sym_degen_df)
                assert list(sym_degen_df.loc[("Int_F", "-1")]) == [
                    regen_expected_site_sym,
                    regen_expected_relax_sym,
                    expected_orientational_degeneracy,
                    1,
                    expected_orientational_degeneracy,
                    expected_site_multiplicity,
                ]


class ReorderedParsingTestCase(unittest.TestCase):
    """
    Test cases where the atoms bulk and defect supercells have been reordered
    with respect to each other, but that site-matching and charge corrections
    are still correctly performed.
    """

    def setUp(self):
        self.module_path = os.path.dirname(os.path.abspath(__file__))
        self.CdTe_corrections_dir = os.path.join(self.module_path, "data/CdTe_charge_correction_tests")
        self.v_Cd_m2_path = f"{self.CdTe_corrections_dir}/v_Cd_-2_vasp_gam"
        self.CdTe_dielectric = np.array([[9.13, 0, 0], [0.0, 9.13, 0], [0, 0, 9.13]])  # CdTe

    @custom_mpl_image_compare(filename="CdTe_v_cd_m2_eigenvalue_plot.png")
    def test_parsing_cdte(self):
        """
        Test parsing CdTe bulk vasp_gam example.
        """
        parsed_v_cd_m2 = defect_entry_from_paths(
            defect_path=self.v_Cd_m2_path,
            bulk_path=f"{self.CdTe_corrections_dir}/bulk_vasp_gam",
            dielectric=self.CdTe_dielectric,
            charge_state=-2,
        )
        uncorrected_energy = 7.4475896
        assert np.isclose(
            parsed_v_cd_m2.get_ediff() - sum(parsed_v_cd_m2.corrections.values()),
            uncorrected_energy,
            atol=1e-3,
        )

        bes, fig = parsed_v_cd_m2.get_eigenvalue_analysis()
        assert not any(
            [
                bes.has_acceptor_phs,
                bes.has_donor_phs,
                bes.has_occupied_localized_state,
                bes.has_unoccupied_localized_state,
                bes.is_shallow,
            ]
        )

        return fig

    def test_kumagai_order(self):
        """
        Test Kumagai defect correction parser can handle mismatched atomic
        orders.
        """
        parsed_v_cd_m2_orig = defect_entry_from_paths(
            defect_path=self.v_Cd_m2_path,
            bulk_path=f"{self.CdTe_corrections_dir}/bulk_vasp_gam",
            dielectric=self.CdTe_dielectric,
            charge_state=-2,
        )
        parsed_v_cd_m2_alt = defect_entry_from_paths(
            defect_path=self.v_Cd_m2_path,
            bulk_path=f"{self.CdTe_corrections_dir}/bulk_vasp_gam_alt",
            dielectric=self.CdTe_dielectric,
            charge_state=-2,
        )
        # should use Kumagai correction by default when OUTCARs available
        assert np.isclose(parsed_v_cd_m2_orig.get_ediff(), parsed_v_cd_m2_alt.get_ediff())
        assert np.isclose(
            sum(parsed_v_cd_m2_orig.corrections.values()), sum(parsed_v_cd_m2_alt.corrections.values())
        )

        # test where the ordering is all over the shop; v_Cd_-2 POSCAR with a Te atom, then 31 randomly
        # ordered Cd atoms, then 31 randomly ordered Te atoms:
        parsed_v_cd_m2_alt2 = defect_entry_from_paths(
            defect_path=f"{self.CdTe_corrections_dir}/v_Cd_-2_choppy_changy_vasp_gam",
            bulk_path=f"{self.CdTe_corrections_dir}/bulk_vasp_gam_alt",
            dielectric=self.CdTe_dielectric,
            charge_state=-2,
        )
        # should use Kumagai correction by default when OUTCARs available
        assert np.isclose(parsed_v_cd_m2_orig.get_ediff(), parsed_v_cd_m2_alt2.get_ediff())
        assert np.isclose(
            sum(parsed_v_cd_m2_orig.corrections.values()), sum(parsed_v_cd_m2_alt2.corrections.values())
        )

    def test_freysoldt_order(self):
        """
        Test Freysoldt defect correction parser can handle mismatched atomic
        orders.
        """
        shutil.move(f"{self.v_Cd_m2_path}/OUTCAR.gz", f"{self.v_Cd_m2_path}/hidden_otcr.gz")  # use FNV
        parsed_v_cd_m2_orig = defect_entry_from_paths(
            defect_path=self.v_Cd_m2_path,
            bulk_path=f"{self.CdTe_corrections_dir}/bulk_vasp_gam",
            dielectric=self.CdTe_dielectric,
            charge_state=-2,
        )
        parsed_v_cd_m2_alt = defect_entry_from_paths(
            defect_path=self.v_Cd_m2_path,
            bulk_path=f"{self.CdTe_corrections_dir}/bulk_vasp_gam_alt",
            dielectric=self.CdTe_dielectric,
            charge_state=-2,
        )
        shutil.move(f"{self.v_Cd_m2_path}/hidden_otcr.gz", f"{self.v_Cd_m2_path}/OUTCAR.gz")  # move back

        # should use Freysoldt correction by default when OUTCARs not available
        assert np.isclose(parsed_v_cd_m2_orig.get_ediff(), parsed_v_cd_m2_alt.get_ediff())
        assert np.isclose(
            sum(parsed_v_cd_m2_orig.corrections.values()), sum(parsed_v_cd_m2_alt.corrections.values())
        )

        # test where the ordering is all over the shop; v_Cd_-2 POSCAR with a Te atom, then 31 randomly
        # ordered Cd atoms, then 31 randomly ordered Te atoms:
        shutil.move(
            f"{self.CdTe_corrections_dir}/v_Cd_-2_choppy_changy_vasp_gam/OUTCAR.gz",
            f"{self.CdTe_corrections_dir}/v_Cd_-2_choppy_changy_vasp_gam/hidden_otcr.gz",
        )  # use FNV
        parsed_v_cd_m2_alt2 = defect_entry_from_paths(
            defect_path=f"{self.CdTe_corrections_dir}/v_Cd_-2_choppy_changy_vasp_gam",
            bulk_path=f"{self.CdTe_corrections_dir}/bulk_vasp_gam",
            dielectric=self.CdTe_dielectric,
            charge_state=-2,
        )
        shutil.move(
            f"{self.CdTe_corrections_dir}/v_Cd_-2_choppy_changy_vasp_gam/hidden_otcr.gz",
            f"{self.CdTe_corrections_dir}/v_Cd_-2_choppy_changy_vasp_gam/OUTCAR.gz",
        )  # move back

        # should use Freysoldt correction by default when OUTCARs not available
        assert np.isclose(parsed_v_cd_m2_orig.get_ediff(), parsed_v_cd_m2_alt2.get_ediff())
        assert np.isclose(
            sum(parsed_v_cd_m2_orig.corrections.values()), sum(parsed_v_cd_m2_alt2.corrections.values())
        )
