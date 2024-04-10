"""
Tests for the `doped.analysis` module, which also implicitly tests most of the
`doped.utils.parsing` module.
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
from pymatgen.core.structure import Structure
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
    get_defect_site_idxs_and_unrelaxed_structure,
    get_defect_type_and_composition_diff,
    get_orientational_degeneracy,
    get_outcar,
    get_procar,
    get_vasprun,
)
from doped.utils.symmetry import point_symmetry

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


class DefectsParsingTestCase(unittest.TestCase):
    def setUp(self):
        self.module_path = os.path.dirname(os.path.abspath(__file__))
        self.EXAMPLE_DIR = os.path.join(self.module_path, "../examples")
        self.CdTe_EXAMPLE_DIR = os.path.abspath(os.path.join(self.module_path, "../examples/CdTe"))
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
        self.Sb2Si2Te6_DATA_DIR = os.path.join(self.EXAMPLE_DIR, "Sb2Si2Te6")

        self.V2O5_DATA_DIR = os.path.join(self.module_path, "data/V2O5")
        self.SrTiO3_DATA_DIR = os.path.join(self.module_path, "data/SrTiO3")
        self.ZnS_DATA_DIR = os.path.join(self.module_path, "data/ZnS")
        self.SOLID_SOLUTION_DATA_DIR = os.path.join(self.module_path, "data/solid_solution")
        self.CaO_DATA_DIR = os.path.join(self.module_path, "data/CaO")
        self.BiOI_DATA_DIR = os.path.join(self.module_path, "data/BiOI")

    def tearDown(self):
        if_present_rm(os.path.join(self.CdTe_BULK_DATA_DIR, "voronoi_nodes.json"))
        if_present_rm(os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_defect_dict.json"))
        if_present_rm(os.path.join(self.CdTe_EXAMPLE_DIR, "test_pop.json"))
        if_present_rm(os.path.join(self.YTOS_EXAMPLE_DIR, "Y2Ti2S2O5_defect_dict.json"))
        if_present_rm(os.path.join(self.Sb2Si2Te6_DATA_DIR, "SiSbTe3_defect_dict.json"))
        if_present_rm(os.path.join(self.Sb2Se3_DATA_DIR, "defect/Sb2Se3_defect_dict.json"))
        if_present_rm("V2O5_test")
        if_present_rm(os.path.join(self.SrTiO3_DATA_DIR, "SrTiO3_defect_dict.json"))
        if_present_rm(os.path.join(self.ZnS_DATA_DIR, "ZnS_defect_dict.json"))
        if_present_rm(os.path.join(self.CaO_DATA_DIR, "CaO_defect_dict.json"))
        if_present_rm(os.path.join(self.BiOI_DATA_DIR, "BiOI_defect_dict.json"))

        for i in os.listdir(self.SOLID_SOLUTION_DATA_DIR):
            if "json" in i:
                if_present_rm(os.path.join(self.SOLID_SOLUTION_DATA_DIR, i))

        for i in os.listdir(f"{self.YTOS_EXAMPLE_DIR}/Bulk"):
            if i.startswith("."):
                if_present_rm(f"{self.YTOS_EXAMPLE_DIR}/Bulk/{i}")

        for i in os.listdir(f"{self.YTOS_EXAMPLE_DIR}/F_O_1"):
            if i.startswith("."):
                if_present_rm(f"{self.YTOS_EXAMPLE_DIR}/F_O_1/{i}")

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
            assert name == defect_entry.name
            if defect_entry.charge_state != 0 and not skip_corrections:
                assert sum(defect_entry.corrections.values()) != 0
            assert defect_entry.get_ediff()  # can get ediff fine
            assert defect_entry.calculation_metadata  # has metadata

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
        assert np.isclose(dp.defect_dict["v_Cd_-2"].get_ediff(), 8.398, atol=1e-3)
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
                    "in the format: (INCAR tag, value in bulk calculation, value in defect",
                    "Int_Te_3_Unperturbed_1: [('ADDGRID', True, False)]",
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

        CdTe_thermo = CdTe_dp.get_defect_thermodynamics(dist_tol=dist_tol)
        dumpfn(
            CdTe_thermo, os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_example_thermo.json")
        )  # for test_plotting
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
            assert CdTe_dp.bulk_band_gap_path is None

        self._check_DefectsParser(CdTe_dp)
        assert (
            os.path.exists(os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_defect_dict.json"))
            or os.path.exists(os.path.join(self.CdTe_EXAMPLE_DIR, "test_pop.json"))  # custom json name
            or os.path.exists(
                os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_example_defect_dict.json")  # custom json name
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

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_DefectsParser_CdTe(self):
        with warnings.catch_warnings(record=True) as w:
            default_dp = DefectsParser(
                output_path=self.CdTe_EXAMPLE_DIR,
                dielectric=9.13,
                json_filename="CdTe_example_defect_dict.json",
            )  # for testing in test_thermodynamics.py
        print([warn.message for warn in w])  # for debugging
        self._check_default_CdTe_DefectsParser_outputs(default_dp, w)  # saves CdTe_example_thermo.json

        # test reloading DefectsParser
        reloaded_defect_dict = loadfn(os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_example_defect_dict.json"))

        for defect_name, defect_entry in reloaded_defect_dict.items():
            assert defect_entry.name == default_dp.defect_dict[defect_name].name
            assert np.isclose(defect_entry.get_ediff(), default_dp.defect_dict[defect_name].get_ediff())
            assert np.allclose(
                defect_entry.sc_defect_frac_coords,
                default_dp.defect_dict[defect_name].sc_defect_frac_coords,
            )

        # integration test using parsed CdTe thermo and chempots for plotting:
        default_thermo = default_dp.get_defect_thermodynamics()

        return default_thermo.plot(chempots=self.CdTe_chempots, limit="CdTe-Te")

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_DefectsParser_CdTe_without_multiprocessing(self):
        shutil.move(  # avoid overwriting
            os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_example_thermo.json"),
            os.path.join(self.CdTe_EXAMPLE_DIR, "orig_CdTe_example_thermo.json"),
        )
        # test same behaviour without multiprocessing:
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(
                output_path=self.CdTe_EXAMPLE_DIR,
                dielectric=9.13,
                processes=1,
                parse_projected_eigen=False,
            )
        print([warn.message for warn in w])  # for debugging
        self._check_default_CdTe_DefectsParser_outputs(dp, w)

        # integration test using parsed CdTe thermo and chempots for plotting:
        default_thermo = dp.get_defect_thermodynamics(chempots=self.CdTe_chempots)
        shutil.move(
            os.path.join(self.CdTe_EXAMPLE_DIR, "orig_CdTe_example_thermo.json"),
            os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_example_thermo.json"),
        )
        return default_thermo.plot(limit="CdTe-Te")

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_DefectsParser_CdTe_filterwarnings(self):
        shutil.move(  # avoid overwriting
            os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_example_thermo.json"),
            os.path.join(self.CdTe_EXAMPLE_DIR, "orig_CdTe_example_thermo.json"),
        )
        # check using filterwarnings works as expected:
        warnings.filterwarnings("ignore", "Multiple")
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(
                output_path=self.CdTe_EXAMPLE_DIR,
                dielectric=9.13,
            )
        print([warn.message for warn in w])  # for debugging
        self._check_default_CdTe_DefectsParser_outputs(dp, w, multiple_outcars_warning=False)
        warnings.filterwarnings("default", "Multiple")

        # integration test using parsed CdTe thermo and chempots for plotting:
        default_thermo = dp.get_defect_thermodynamics(chempots=self.CdTe_chempots)
        shutil.move(
            os.path.join(self.CdTe_EXAMPLE_DIR, "orig_CdTe_example_thermo.json"),
            os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_example_thermo.json"),
        )
        return default_thermo.plot(limit="CdTe-Te")

    def test_DefectsParser_CdTe_dist_tol(self):
        shutil.move(  # avoid overwriting
            os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_example_thermo.json"),
            os.path.join(self.CdTe_EXAMPLE_DIR, "orig_CdTe_example_thermo.json"),
        )
        # test with reduced dist_tol:
        # Int_Te_3_Unperturbed merged with Int_Te_3 with default dist_tol = 1.5, now no longer merged
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(
                output_path=self.CdTe_EXAMPLE_DIR, dielectric=9.13, parse_projected_eigen=False
            )
        print([warn.message for warn in w])  # for debugging
        self._check_default_CdTe_DefectsParser_outputs(dp, w, dist_tol=0.1)
        shutil.move(
            os.path.join(self.CdTe_EXAMPLE_DIR, "orig_CdTe_example_thermo.json"),
            os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_example_thermo.json"),
        )

    @custom_mpl_image_compare(filename="CdTe_Te_Cd_+1_eigenvalue_plot.png")
    def test_DefectsParser_CdTe_no_dielectric_json(self):
        # test no dielectric and no JSON:
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(
                output_path=self.CdTe_EXAMPLE_DIR,
                json_filename=False,
            )
        print([warn.message for warn in w])  # for debugging
        assert any(
            "The dielectric constant (`dielectric`) is needed to compute finite-size charge "
            "corrections, but none was provided" in str(warn.message)
            for warn in w
        )
        self._check_DefectsParser(dp, skip_corrections=True)
        assert not os.path.exists(os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_defect_dict.json"))

        bes, fig = dp.defect_dict["Te_Cd_+1"].get_eigenvalue_analysis(plot=True)
        assert bes.has_unoccupied_localized_state  # has in-gap hole polaron state
        assert not any(
            [bes.has_acceptor_phs, bes.has_donor_phs, bes.has_occupied_localized_state, bes.is_shallow]
        )

        return fig

    def test_DefectsParser_CdTe_custom_settings(self):
        shutil.move(  # avoid overwriting
            os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_example_thermo.json"),
            os.path.join(self.CdTe_EXAMPLE_DIR, "orig_CdTe_example_thermo.json"),
        )

        # test custom settings:
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(
                output_path=self.CdTe_EXAMPLE_DIR,
                dielectric=[9.13, 9.13, 9.13],
                error_tolerance=0.01,
                skip_corrections=False,
                bulk_band_gap_path=self.CdTe_BULK_DATA_DIR,
                processes=4,
                json_filename="test_pop.json",
            )
        print([warn.message for warn in w])  # for debugging
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

        shutil.move(
            os.path.join(self.CdTe_EXAMPLE_DIR, "orig_CdTe_example_thermo.json"),
            os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_example_thermo.json"),
        )

        # test changed attributes:
        assert dp.output_path == self.CdTe_EXAMPLE_DIR
        assert dp.dielectric == [9.13, 9.13, 9.13]
        assert dp.error_tolerance == 0.01
        assert dp.bulk_band_gap_path == self.CdTe_BULK_DATA_DIR
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
            f"{self.CdTe_EXAMPLE_DIR}/CdTe_bulk or subfolder: vasp_std - please ensure `vasprun.xml(.gz)` "
            f"files are present and/or specify `bulk_path` manually."
        ) in str(exc.value)

    def test_DefectsParser_CdTe_skip_corrections(self):
        # skip_corrections:
        dp = DefectsParser(
            output_path=self.CdTe_EXAMPLE_DIR, skip_corrections=True, parse_projected_eigen=False
        )
        self._check_DefectsParser(dp, skip_corrections=True)

    def test_DefectsParser_CdTe_aniso_dielectric(self):
        # anisotropic dielectric
        fake_aniso_dielectric = [1, 2, 3]
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(
                output_path=self.CdTe_EXAMPLE_DIR,
                dielectric=fake_aniso_dielectric,
            )
        print([warn.message for warn in w])  # for debugging
        assert any(
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
            "Defects: ['v_Cd_-2', 'v_Cd_-1'] each encountered the same warning:" in str(warn.message)
            for warn in w
        ) or any(
            "Defects: ['v_Cd_-1', 'v_Cd_-2'] each encountered the same warning:" in str(warn.message)
            for warn in w
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
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(
                output_path=self.CdTe_EXAMPLE_DIR,
                bulk_path=f"{self.module_path}/data/CdTe",  # vasp_gam bulk vr here
                dielectric=9.13,
                parse_projected_eigen=False,
            )
        print([warn.message for warn in w])  # for debugging
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
            "There are mismatching INCAR tags for (some of) your bulk and defect calculations",
            "There are mismatching KPOINTS for (some of) your bulk and defect calculations ",
            "Found the following differences:",
            "(in the format: (bulk kpoints, defect kpoints)):",
            "Int_Te_3_1: [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], "
            "[0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]]]",
            "v_Cd_0: [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5],",
            "Int_Te_3_Unperturbed_1: [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], ",
            "In general, the same KPOINTS settings should be used",
        ]:
            assert any(i in str(warn.message) for warn in w)
        dp.get_defect_thermodynamics()  # test thermo generation works fine

    def test_DefectsParser_corrections_errors_warning(self):
        with warnings.catch_warnings(record=True) as w:
            DefectsParser(
                output_path=self.CdTe_EXAMPLE_DIR,
                dielectric=9.13,
                error_tolerance=0.001,
            )  # low error tolerance to force warnings
        print([warn.message for warn in w])  # for debugging

        assert all(
            any(i in str(warn.message) for warn in w)
            for i in [
                "Estimated error in the Freysoldt (FNV) ",
                "Estimated error in the Kumagai (eFNV) ",
                "charge correction for certain defects is greater than the `error_tolerance` (= "
                "1.00e-03 eV):",
                "v_Cd_-2: 1.13e-02 eV",
                "v_Cd_-1: 7.91e-03 eV",
                "Int_Te_3_1: 3.10e-03 eV",
                "Te_Cd_+1: 2.02e-03 eV",
                "Int_Te_3_Unperturbed_1: 4.91e-03 eV",
                "Int_Te_3_2: 1.24e-02 eV",
                "You may want to check the accuracy of the corrections by",
                "(using `defect_entry.get_freysoldt_correction()` with `plot=True`)",
                "(using `defect_entry.get_kumagai_correction()` with `plot=True`)",
            ]
        )  # correction errors warnings

    @custom_mpl_image_compare(filename="YTOS_example_defects_plot.png")
    def test_DefectsParser_YTOS_default_bulk(self):
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(
                output_path=self.YTOS_EXAMPLE_DIR,
                dielectric=self.ytos_dielectric,
                json_filename="YTOS_example_defect_dict.json",
            )  # for testing in test_thermodynamics.py
        print([warn.message for warn in w])  # for debugging
        assert not w
        self._check_DefectsParser(dp)
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

        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(
                output_path=self.YTOS_EXAMPLE_DIR,
                dielectric=self.ytos_dielectric,
                json_filename="YTOS_example_defect_dict.json",
            )  # for testing in test_thermodynamics.py
        print([warn.message for warn in w])  # for debugging
        assert not w  # hidden files ignored
        self._check_DefectsParser(dp)
        thermo = dp.get_defect_thermodynamics()
        dumpfn(
            thermo, os.path.join(self.YTOS_EXAMPLE_DIR, "YTOS_example_thermo.json")
        )  # for test_plotting
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
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(
                output_path=self.YTOS_EXAMPLE_DIR,
                dielectric=self.ytos_dielectric,
                json_filename="YTOS_example_defect_dict.json",
                parse_projected_eigen=False,
            )  # for testing in test_thermodynamics.py
        print([warn.message for warn in w])  # for debugging
        assert not w  # hidden files ignored
        self._check_DefectsParser(dp)
        thermo = dp.get_defect_thermodynamics()
        dumpfn(
            thermo, os.path.join(self.YTOS_EXAMPLE_DIR, "YTOS_example_thermo.json")
        )  # for test_plotting
        return thermo.plot()  # no chempots for YTOS formation energy plot test

    @custom_mpl_image_compare(filename="YTOS_example_defects_plot.png")
    def test_DefectsParser_YTOS_explicit_bulk(self):
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(
                output_path=self.YTOS_EXAMPLE_DIR,
                bulk_path=os.path.join(self.YTOS_EXAMPLE_DIR, "Bulk"),
                dielectric=self.ytos_dielectric,
                parse_projected_eigen=False,
            )
        print([warn.message for warn in w])  # for debugging
        assert not w
        self._check_DefectsParser(dp)
        thermo = dp.get_defect_thermodynamics()
        dumpfn(
            thermo, os.path.join(self.YTOS_EXAMPLE_DIR, "YTOS_example_thermo.json")
        )  # for test_plotting
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

        with warnings.catch_warnings(record=True) as w:  # no warning about negative corrections with
            # strong anisotropic dielectric:
            Sb2Se3_O_dp = DefectsParser(
                output_path=f"{self.Sb2Se3_DATA_DIR}/defect",
                bulk_path=f"{self.Sb2Se3_DATA_DIR}/bulk",
                dielectric=self.Sb2Se3_dielectric,
                json_filename="Sb2Se3_O_example_defect_dict.json",
            )  # for testing in test_thermodynamics.py
        print([warn.message for warn in w])  # for debugging
        assert not w  # no warnings
        self._check_DefectsParser(Sb2Se3_O_dp)
        Sb2Se3_O_thermo = Sb2Se3_O_dp.get_defect_thermodynamics()
        dumpfn(Sb2Se3_O_thermo, os.path.join(self.Sb2Se3_DATA_DIR, "Sb2Se3_O_example_thermo.json"))  # for
        # test_plotting

        # warning about negative corrections when using (fake) isotropic dielectric:
        with warnings.catch_warnings(record=True) as w:
            Sb2Se3_O_dp = DefectsParser(
                output_path=f"{self.Sb2Se3_DATA_DIR}/defect",
                bulk_path=f"{self.Sb2Se3_DATA_DIR}/bulk",
                dielectric=40,  # fake isotropic dielectric
                parse_projected_eigen=False,
            )
        print([warn.message for warn in w])  # for debugging
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

        return Sb2Se3_O_thermo.plot(chempots={"O": -8.9052, "Se": -5})  # example chempots

    @custom_mpl_image_compare(filename="Sb2Si2Te6_v_Sb_-3_eFNV_plot_no_intralayer.png")
    def test_sb2si2te6_eFNV(self):
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(
                self.Sb2Si2Te6_DATA_DIR,
                dielectric=self.Sb2Si2Te6_dielectric,
                json_filename="Sb2Si2Te6_example_defect_dict.json",  # testing in test_thermodynamics.py
                parse_projected_eigen=False,
            )
        print([str(warning.message) for warning in w])  # for debugging
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
        dumpfn(sb2si2te6_thermo, os.path.join(self.Sb2Si2Te6_DATA_DIR, "Sb2Si2Te6_example_thermo.json"))
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
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(
                self.V2O5_DATA_DIR,
                dielectric=[4.186, 19.33, 17.49],
                json_filename="V2O5_example_defect_dict.json",  # testing in test_thermodynamics.py
            )
        print([str(warning.message) for warning in w])  # for debugging
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

        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser("V2O5_test", dielectric=[4.186, 19.33, 17.49])
        print([str(warning.message) for warning in w])  # for debugging
        assert not w  # no warnings
        assert len(dp.defect_dict) == 5  # now 5 defects, all still included
        self._check_DefectsParser(dp)
        thermo = dp.get_defect_thermodynamics()
        v2o5_chempots = loadfn(os.path.join(self.V2O5_DATA_DIR, "chempots.json"))
        thermo.chempots = v2o5_chempots

        print(thermo.get_symmetries_and_degeneracies())

        return thermo.plot(limit="V2O5-O2")

    @custom_mpl_image_compare(filename="SrTiO3_v_O.png")
    def test_SrTiO3_diff_ISYM_bulk_defect(self):
        """
        Test parsing SrTiO3 defect calculations, where a different ISYM was
        used for the bulk (= 3) compared to the defect (= 0) calculations.

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
                "[('LASPH', False, True)]",
            ]
        )

        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(
                self.SrTiO3_DATA_DIR, dielectric=6.33, parse_projected_eigen=False
            )  # wrong dielectric from Kanta
        print([str(warning.message) for warning in w])  # for debugging
        assert len(w) == 1
        assert all(
            i in str(w[0].message)
            for i in [
                "There are mismatching INCAR tags",
                "vac_O_0: [('LASPH', False, True)]",
                "vac_O_1: [('LASPH', False, True)]",
                "vac_O_2: [('LASPH', False, True)]",
            ]
        )

        assert len(dp.defect_dict) == 3
        self._check_DefectsParser(dp)

        # some hardcoded symmetry tests with default `symprec = 0.1` for relaxed structures:
        assert dp.defect_dict["vac_O_2"].calculation_metadata["relaxed point symmetry"] == "C2v"
        assert dp.defect_dict["vac_O_1"].calculation_metadata["relaxed point symmetry"] == "Cs"
        assert dp.defect_dict["vac_O_0"].calculation_metadata["relaxed point symmetry"] == "C2v"
        thermo = dp.get_defect_thermodynamics()

        print(thermo.get_symmetries_and_degeneracies())

        return thermo.plot()

    @custom_mpl_image_compare(filename="ZnS_defects.png")
    def test_ZnS_non_diagonal_NKRED_mismatch(self):
        """
        Test parsing ZnS defect calculations, which were performed with a non-
        diagonal periodicity-breaking supercell, and with NKRED mismatch from
        defect and bulk supercells.
        """
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(self.ZnS_DATA_DIR, dielectric=8.9)
        print([str(warning.message) for warning in w])  # for debugging
        assert len(w) == 1
        assert all(
            i in str(w[0].message)
            for i in [
                "There are mismatching INCAR tags",
                "vac_1_Zn_0: [('NKRED', 2, 1)]",
                "vac_1_Zn_-2: [('NKRED', 2, 1)]",
                "vac_2_S_2: [('NKRED', 2, 1)]",
                "inter_29_Al_3: [('NKRED', 2, 1)]",
                "sub_1_Al_on_Zn_-1: [('NKRED', 2, 1)]",
            ]
        )

        assert len(dp.defect_dict) == 17
        self._check_DefectsParser(dp)
        thermo = dp.get_defect_thermodynamics()

        with warnings.catch_warnings(record=True) as w:
            symm_df = thermo.get_symmetries_and_degeneracies()
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
            symm_df["Defect"].str.contains("vac", na=False)
            | symm_df["Defect"].str.contains("sub", na=False)
        ]
        assert list(vacancy_and_sub_rows["Site_Symm"].unique()) == ["Td"]
        assert list(vacancy_and_sub_rows["Defect_Symm"].unique()) == ["C1"]

        interstitial_rows = symm_df[symm_df["Defect"].str.contains("inter", na=False)]
        assert list(interstitial_rows["Site_Symm"].unique()) == ["C3v", "Cs", "C1"]
        assert list(interstitial_rows["Defect_Symm"].unique()) == ["C1"]

        thermo.dist_tol = 2.5  # merges Al interstitials together
        thermo.to_json(os.path.join(self.ZnS_DATA_DIR, "ZnS_thermo.json"))
        return thermo.plot()

    def test_solid_solution_oxi_state_handling(self):
        """
        Test parsing a defect in a large, complex solid solution supercell,
        which hangs with using ``pymatgen``'s oxi state methods (so is set as
        'undetermined' by ``doped``, as this property isn't necessary when
        parsing).
        """
        with warnings.catch_warnings(record=True) as w:
            # no warning with no dielectric/OUTCARs, as is neutral
            dp = DefectsParser(self.SOLID_SOLUTION_DATA_DIR, parse_projected_eigen=False)
        print([str(warning.message) for warning in w])  # for debugging
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
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(
                output_path=self.CaO_DATA_DIR,
                skip_corrections=True,
            )

        print([str(warning.message) for warning in w])  # for debugging
        assert not w
        assert len(dp.defect_dict) == 4
        self._check_DefectsParser(dp, skip_corrections=True)

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
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(
                output_path=self.BiOI_DATA_DIR,
                skip_corrections=True,
            )

        print([str(warning.message) for warning in w])  # for debugging
        assert not w
        assert len(dp.defect_dict) == 1
        self._check_DefectsParser(dp, skip_corrections=True)

        # some hardcoded symmetry tests with default `symprec = 0.1` for relaxed structures:
        assert dp.defect_dict["v_Bi_+1"].calculation_metadata["bulk site symmetry"] == "C4v"
        assert dp.defect_dict["v_Bi_+1"].calculation_metadata["relaxed point symmetry"] == "Cs"

        # test setting symprec during parsing
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(
                output_path=self.BiOI_DATA_DIR,
                skip_corrections=True,
                symprec=0.01,
                parse_projected_eigen=False,
            )

        print([str(warning.message) for warning in w])  # for debugging
        assert not w
        assert len(dp.defect_dict) == 1
        self._check_DefectsParser(dp, skip_corrections=True)

        # some hardcoded symmetry tests with default `symprec = 0.1` for relaxed structures:
        assert dp.defect_dict["v_Bi_+1"].calculation_metadata["bulk site symmetry"] == "C4v"
        assert dp.defect_dict["v_Bi_+1"].calculation_metadata["relaxed point symmetry"] == "C1"

        assert get_orientational_degeneracy(dp.defect_dict["v_Bi_+1"]) == 4.0
        assert get_orientational_degeneracy(dp.defect_dict["v_Bi_+1"], symprec=0.01) == 8.0


class DopedParsingTestCase(unittest.TestCase):
    def setUp(self):
        self.module_path = os.path.dirname(os.path.abspath(__file__))
        self.EXAMPLE_DIR = os.path.join(self.module_path, "../examples")
        self.CdTe_EXAMPLE_DIR = os.path.abspath(os.path.join(self.module_path, "../examples/CdTe"))
        self.YTOS_EXAMPLE_DIR = os.path.join(self.module_path, "../examples/YTOS")
        self.CdTe_BULK_DATA_DIR = os.path.abspath(
            os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_bulk/vasp_ncl")
        )
        self.CdTe_dielectric = np.array([[9.13, 0, 0], [0.0, 9.13, 0], [0, 0, 9.13]])  # CdTe

        self.ytos_dielectric = [  # from legacy Materials Project
            [40.71948719643814, -9.282128210266565e-14, 1.26076160303219e-14],
            [-9.301652644020242e-14, 40.71948719776858, 4.149879443489052e-14],
            [5.311743673463141e-15, 2.041077680836527e-14, 25.237620491130023],
        ]

        self.Sb2Se3_DATA_DIR = os.path.join(self.module_path, "data/Sb2Se3")
        self.Sb2Se3_dielectric = np.array([[85.64, 0, 0], [0.0, 128.18, 0], [0, 0, 15.00]])

    def tearDown(self):
        if_present_rm(os.path.join(self.CdTe_BULK_DATA_DIR, "voronoi_nodes.json"))

        if os.path.exists(f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/hidden_otcr.gz"):
            shutil.move(
                f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/hidden_otcr.gz",
                f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/OUTCAR.gz",
            )

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

        if_present_rm(f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/LOCPOT.gz")

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
                parse_projected_eigen=False,
            )
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert (
                "Auto-determined defect charge q=-2 does not match specified charge q=-1. Will continue "
                "with specified charge_state, but beware!" in str(w[-1].message)
            )
            assert np.isclose(
                parsed_v_cd_m1.corrections["freysoldt_charge_correction"], 0.26066457692529815
            )

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
        """
        defect_path = f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl"
        fake_aniso_dielectric = [1, 2, 3]

        with warnings.catch_warnings(record=True) as w:
            parsed_v_cd_m2_fake_aniso_dp = DefectParser.from_paths(
                defect_path=defect_path,
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=fake_aniso_dielectric,
                parse_projected_eigen=False,
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
                parse_projected_eigen=False,
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
                f"https://doped.readthedocs.io/en/latest/Tips.html#perturbed-host-states). "
                f"If this error is not acceptable, you may need to use a larger supercell for more "
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
            parsed_int_Te_2_fake_aniso.get_ediff(), -4.7620, atol=1e-3
        )  # -4.734 with old voronoi frac coords

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
                parse_projected_eigen=False,
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
            f"`ICORELEVEL` was not set to 0 (= default) in the `INCAR`, or if the calculation was "
            f"finished prematurely with a `STOPCAR`. The Kumagai charge correction cannot be computed "
            f"without this data!\n{action}" in str(warnings[0].message)
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
            parse_projected_eigen=False,
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
                parse_projected_eigen=False,
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
                parse_projected_eigen=False,
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
            parse_projected_eigen=False,
        )

        # Check that the correct Freysoldt correction is applied
        correct_correction_dict = {
            "freysoldt_charge_correction": 0.7376460317828045,
        }
        for correction_name, correction_energy in correct_correction_dict.items():
            assert np.isclose(
                parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                atol=1e-3,
            )

        # test float
        new_parsed_v_cd_m2 = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=self.CdTe_BULK_DATA_DIR,
            dielectric=9.13,
            parse_projected_eigen=False,
        )
        for correction_name, correction_energy in correct_correction_dict.items():
            assert np.isclose(
                new_parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                atol=1e-3,
            )

        # test int
        new_parsed_v_cd_m2 = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=self.CdTe_BULK_DATA_DIR,
            dielectric=9,
            charge_state=-2,
            parse_projected_eigen=False,
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
            parse_projected_eigen=False,
        )
        for correction_name, correction_energy in correct_correction_dict.items():
            assert np.isclose(
                new_parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                atol=1e-3,
            )

        # test 3x1 list
        new_parsed_v_cd_m2 = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=self.CdTe_BULK_DATA_DIR,
            dielectric=[9.13, 9.13, 9.13],
            charge_state=-2,
            parse_projected_eigen=False,
        )
        for correction_name, correction_energy in correct_correction_dict.items():
            assert np.isclose(
                new_parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                atol=1e-3,
            )

        # test 3x3 array
        new_parsed_v_cd_m2 = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=self.CdTe_BULK_DATA_DIR,
            dielectric=self.CdTe_dielectric,
            parse_projected_eigen=False,
        )
        for correction_name, correction_energy in correct_correction_dict.items():
            assert np.isclose(
                new_parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                atol=1e-3,
            )

        # test 3x3 list
        new_parsed_v_cd_m2 = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=self.CdTe_BULK_DATA_DIR,
            dielectric=self.CdTe_dielectric.tolist(),
            charge_state=-2,
            parse_projected_eigen=False,
        )
        for correction_name, correction_energy in correct_correction_dict.items():
            assert np.isclose(
                new_parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                atol=1e-3,
            )

    def test_vacancy_parsing_and_freysoldt(self):
        """
        Test parsing of Cd vacancy calculations and correct Freysoldt
        correction calculated.
        """
        parsed_vac_Cd_dict = {}

        for i in os.listdir(self.CdTe_EXAMPLE_DIR):
            # loop folders and parse those with "v_Cd" in name:
            if os.path.isdir(f"{self.CdTe_EXAMPLE_DIR}/{i}") and "v_Cd" in i:
                defect_path = f"{self.CdTe_EXAMPLE_DIR}/{i}/vasp_ncl"
                int(i[-2:].replace("_", ""))
                # parse with no transformation.json
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
            assert np.isclose(parsed_vac_Cd_dict[name].get_ediff(), energy, atol=1e-3)
            for correction_name, correction_energy in correction_dict.items():
                assert np.isclose(
                    parsed_vac_Cd_dict[name].corrections[correction_name],
                    correction_energy,
                    atol=1e-3,
                )

            # assert auto-determined vacancy site is correct
            # should be: PeriodicSite: Cd (6.5434, 6.5434, 6.5434) [0.5000, 0.5000, 0.5000]
            if name == "v_Cd_0":
                np.testing.assert_array_almost_equal(
                    parsed_vac_Cd_dict[name].defect_supercell_site.frac_coords, [0.5, 0.5, 0.5]
                )
            else:
                np.testing.assert_array_almost_equal(
                    parsed_vac_Cd_dict[name].defect_supercell_site.frac_coords, [0, 0, 0]
                )

    def test_interstitial_parsing_and_kumagai(self):
        """
        Test parsing of Te (split-)interstitial and Kumagai-Oba (eFNV)
        correction.
        """
        if_present_rm(os.path.join(self.CdTe_BULK_DATA_DIR, "voronoi_nodes.json"))
        with patch("builtins.print") as mock_print:
            te_i_2_ent = defect_entry_from_paths(
                defect_path=f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl",
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=self.CdTe_dielectric,
                charge_state=+2,  # test manually specifying charge state
                parse_projected_eigen=False,
            )

        self._check_defect_entry_corrections(te_i_2_ent, -6.2009, 0.9038318161163628)
        # assert auto-determined interstitial site is correct
        # initial position is: PeriodicSite: Te (12.2688, 12.2688, 8.9972) [0.9375, 0.9375, 0.6875]
        np.testing.assert_array_almost_equal(
            te_i_2_ent.defect_supercell_site.frac_coords, [0.834511, 0.943944, 0.69776]
        )

        # run again to check parsing of previous Voronoi sites
        with patch("builtins.print") as mock_print:
            te_i_2_ent = defect_entry_from_paths(
                defect_path=f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl",
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=self.CdTe_dielectric,
                charge_state=+2,  # test manually specifying charge state
                parse_projected_eigen=False,
            )

        mock_print.assert_not_called()
        if_present_rm(os.path.join(self.CdTe_BULK_DATA_DIR, "voronoi_nodes.json"))

    def test_substitution_parsing_and_kumagai(self):
        """
        Test parsing of Te_Cd_1 and Kumagai-Oba (eFNV) correction.
        """
        for i in os.listdir(self.CdTe_EXAMPLE_DIR):
            if "Te_Cd" in i:  # loop folders and parse those with "Te_Cd" in name
                defect_path = f"{self.CdTe_EXAMPLE_DIR}/{i}/vasp_ncl"
                defect_charge = int(i[-2:].replace("_", ""))
                # parse with no transformation.json:
                te_cd_1_ent = defect_entry_from_paths(
                    defect_path=defect_path,
                    bulk_path=self.CdTe_BULK_DATA_DIR,
                    dielectric=self.CdTe_dielectric,
                    charge_state=defect_charge,
                    parse_projected_eigen=False,
                )

        self._check_defect_entry_corrections(te_cd_1_ent, -2.6676, 0.23840982963691623)
        # assert auto-determined substitution site is correct
        # should be: PeriodicSite: Te (6.5434, 6.5434, 6.5434) [0.5000, 0.5000, 0.5000]
        np.testing.assert_array_almost_equal(
            te_cd_1_ent.defect_supercell_site.frac_coords, [0.475139, 0.475137, 0.524856]
        )

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
            bulk_site_idx,
            defect_site_idx,
            unrelaxed_defect_structure,
        ) = get_defect_site_idxs_and_unrelaxed_structure(
            bulk_sc_structure, initial_defect_structure, def_type, comp_diff
        )
        assert bulk_site_idx is None
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
            bulk_site_idx,
            defect_site_idx,
            unrelaxed_defect_structure,
        ) = get_defect_site_idxs_and_unrelaxed_structure(
            bulk_sc_structure, initial_defect_structure, def_type, comp_diff
        )
        assert bulk_site_idx == 0
        assert defect_site_idx == 63  # last site in structure

        # assert auto-determined substitution site is correct
        np.testing.assert_array_almost_equal(
            unrelaxed_defect_structure[defect_site_idx].frac_coords,
            [0.00, 0.00, 0.00],
            decimal=2,  # exact match because perfect supercell
        )

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

        # parse with no transformation.json or explicitly-set-charge:
        with warnings.catch_warnings(record=True) as w:
            int_F_minus1_ent = defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk/",
                dielectric=self.ytos_dielectric,
                parse_projected_eigen=False,
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

        if_present_rm(os.path.join(self.YTOS_EXAMPLE_DIR, "Bulk", "voronoi_nodes.json"))

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
            f"https://doped.readthedocs.io/en/latest/Tips.html#perturbed-host-states). If this error is "
            f"not acceptable, you may need to use a larger supercell for more accurate energies."
            in str(w[0].message)
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
            assert not w  # this supercell is not periodicity breaking
        assert relaxed_defect_name == "F_i_C4v_O2.67"
        assert get_defect_name_from_entry(int_F_minus1_ent, relaxed=False) == "F_i_Cs_O2.67"

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

        # parse with no transformation.json or explicitly-set-charge:
        with warnings.catch_warnings(record=True) as w:
            F_O_1_ent = defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk/",
                dielectric=self.ytos_dielectric,
                parse_projected_eigen=False,
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
                parse_projected_eigen=False,
            )  # check no correction error warning with default tolerance:
        print([str(warn.message) for warn in w])
        assert any(
            f"Estimated error in the Freysoldt (FNV) charge correction for defect {F_O_1_ent.name} is "
            f"0.000 eV (i.e. which is greater than the `error_tolerance`: 0.000 eV). You may want to "
            f"check the accuracy of the correction by plotting the site potential differences (using "
            f"`defect_entry.get_freysoldt_correction()` with `plot=True`). Large errors are often due "
            f"to unstable or shallow defect charge states (which can't be accurately modelled with "
            f"the supercell approach; see "
            f"https://doped.readthedocs.io/en/latest/Tips.html#perturbed-host-states). "
            f"If this error is not acceptable, you may need to use a larger supercell for more "
            f"accurate energies." in str(warning.message)
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
        # parse with no transformation.json or explicitly-set-charge:
        F_O_1_ent = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk/",
            dielectric=self.ytos_dielectric,
            charge_state=1,
            parse_projected_eigen=False,
        )

        self._test_F_O_1_ent(F_O_1_ent, 0.04176, "kumagai_charge_correction", 0.12699488572686776)

        # test symmetry determination (no warning here because periodicity breaking doesn't affect F_O):
        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            relaxed_defect_name = get_defect_name_from_entry(F_O_1_ent)
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

    def test_voronoi_structure_mismatch_and_reparse(self):
        """
        Test that a mismatch in bulk_supercell structure from previously parsed
        Voronoi nodes json file with current defect bulk supercell is detected
        and re-parsed.
        """
        with patch("builtins.print"):
            for i in os.listdir(self.CdTe_EXAMPLE_DIR):
                if "Int_Te" in i:  # loop folders and parse those with "Int_Te" in name
                    defect_path = f"{self.CdTe_EXAMPLE_DIR}/{i}/vasp_ncl"
                    # parse with no transformation.json or explicitly-set-charge:
                    defect_entry_from_paths(
                        defect_path=defect_path,
                        bulk_path=self.CdTe_BULK_DATA_DIR,
                        dielectric=self.CdTe_dielectric,
                        parse_projected_eigen=False,
                    )
        shutil.copyfile(
            os.path.join(self.CdTe_BULK_DATA_DIR, "voronoi_nodes.json"),
            f"{self.YTOS_EXAMPLE_DIR}/Bulk/voronoi_nodes.json",
        )  # mismatching voronoi nodes

        with warnings.catch_warnings(record=True) as w:
            defect_path = f"{self.YTOS_EXAMPLE_DIR}/Int_F_-1/"
            # parse with no transformation.json or explicitly-set-charge:
            defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk/",
                dielectric=self.ytos_dielectric,
                charge_state=-1,  # test manually specifying charge state
                parse_projected_eigen=False,
            )

        warning_message = (
            "Previous bulk voronoi_nodes.json detected, but does not match current bulk supercell. "
            "Recalculating Voronoi nodes."
        )
        user_warnings = [warning for warning in w if warning.category == UserWarning]
        assert len(user_warnings) == 1
        assert warning_message in str(user_warnings[0].message)
        if_present_rm(os.path.join(self.YTOS_EXAMPLE_DIR, "Bulk", "voronoi_nodes.json"))

    def test_tricky_relaxed_interstitial_corrections_kumagai(self):
        """
        Test the eFNV correction performance with tricky-to-locate relaxed
        interstitial sites.

        In this test case, we look at Te_i^+1 ground-state and metastable
        structures from Kavanagh et al. 2022 doi.org/10.1039/D2FD00043A.
        """
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
                parse_projected_eigen=False,
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

        # parse with no transformation.json or explicitly-set-charge:
        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk/",
                parse_projected_eigen=False,
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


class DopedParsingFunctionsTestCase(unittest.TestCase):
    def setUp(self):
        DopedParsingTestCase.setUp(self)  # get attributes from DopedParsingTestCase
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.prim_cdte = Structure.from_file(f"{self.EXAMPLE_DIR}/CdTe/relaxed_primitive_POSCAR")
        self.ytos_bulk_supercell = Structure.from_file(f"{self.EXAMPLE_DIR}/YTOS/Bulk/POSCAR")
        self.lmno_primitive = Structure.from_file(f"{self.data_dir}/Li2Mn3NiO8_POSCAR")
        self.non_diagonal_ZnS = Structure.from_file(f"{self.data_dir}/non_diagonal_ZnS_supercell_POSCAR")
        self.ZnS_DATA_DIR = os.path.join(self.module_path, "data/ZnS")
        self.Cu2SiSe3_EXAMPLE_DIR = os.path.join(self.module_path, "../examples/Cu2SiSe3")
        self.MgO_EXAMPLE_DIR = os.path.join(self.module_path, "../examples/MgO")
        self.Sb2Se3_DATA_DIR = os.path.join(self.module_path, "data/Sb2Se3")
        self.CdTe_EXAMPLE_DIR = os.path.join(self.module_path, "../examples/CdTe")

    def tearDown(self):
        if_present_rm(os.path.join(self.CdTe_BULK_DATA_DIR, "voronoi_nodes.json"))
        if_present_rm(os.path.join(self.YTOS_EXAMPLE_DIR, "Bulk", "voronoi_nodes.json"))
        if_present_rm("./vasprun.xml")

        for dir in ["bulk", "v_Cu_0", "Si_i_-1"]:
            if os.path.exists(f"{self.Cu2SiSe3_EXAMPLE_DIR}/{dir}/vasp_std/hidden_vr.gz"):
                shutil.move(
                    f"{self.Cu2SiSe3_EXAMPLE_DIR}/{dir}/vasp_std/hidden_vr.gz",
                    f"{self.Cu2SiSe3_EXAMPLE_DIR}/{dir}/vasp_std/vasprun.xml.gz",
                )

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
        v_Zn_0 = next(
            defect_entry
            for defect_entry in zns_defect_thermo.defect_entries
            if defect_entry.name == "vac_1_Zn_0"
        )
        Al_Zn_m1 = next(
            defect_entry
            for defect_entry in zns_defect_thermo.defect_entries
            if defect_entry.name == "sub_1_Al_on_Zn_-1"
        )
        Al_i_2 = next(
            defect_entry
            for defect_entry in zns_defect_thermo.defect_entries
            if defect_entry.name == "inter_26_Al_2"
        )

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
                        bulk_voronoi_node_dict,
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
                    assert np.allclose(
                        defect.site.frac_coords, defect_entry.defect_supercell_site.frac_coords, atol=1e-2
                    )
                    assert (
                        unrelaxed_defect_structure
                        == defect_entry.calculation_metadata["unrelaxed_defect_structure"]
                    )
                else:  # interstitial
                    assert np.allclose(defect.site.frac_coords, rattled_relaxed_defect_coords)
                    assert np.allclose(
                        defect_site_in_bulk.frac_coords,
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
                        bulk_voronoi_node_dict,
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
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(self.ZnS_DATA_DIR, dielectric=8.9)
        print([str(warning.message) for warning in w])  # for debugging
        assert len(dp.defect_dict) == 17

        with warnings.catch_warnings(record=True) as w:
            point_symm, periodicity_breaking = point_symmetry(
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
                assert point_symmetry(defect_entry.defect_supercell) == "C1"
            print([str(warning.message) for warning in w])  # for debugging
            assert not w  # no warnings with just defect supercell as can't determine periodicity breaking
            with warnings.catch_warnings(record=True) as w:
                assert point_symmetry(
                    defect_entry.defect_supercell, defect_entry.bulk_supercell, relaxed=False
                ) in ["Td", "C3v", "Cs", "C1"]
            print([str(warning.message) for warning in w])  # for debugging
            assert not w  # no periodicity breaking warning with `relaxed=False`
            with pytest.raises(RuntimeError) as excinfo:
                point_symmetry(defect_entry.defect_supercell, relaxed=False)
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
                parse_projected_eigen=False,
            )
            assert len(w) == 1
            assert all(
                i in str(w[-1].message)
                for i in [
                    "There are mismatching INCAR tags for your bulk and defect calculations",
                    "[('ADDGRID', True, False)]",
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
                parse_projected_eigen=False,
            )
            assert len(w) == 1
            assert all(
                i in str(w[-1].message)
                for i in [
                    "There are mismatching INCAR tags for your bulk and defect calculations",
                    "[('ADDGRID', True, False), ('ENCUT', 450.0, 500.0)]",
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
                parse_projected_eigen=False,
            )
            assert len(w) == 2  # now INCAR and KPOINTS warnings!
            assert any(
                all(
                    i in str(warning.message)
                    for i in [
                        "The KPOINTS for your bulk and defect calculations do not match",
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
                parse_projected_eigen=False,
            )
            assert len(w) == 3  # now INCAR and KPOINTS and POTCAR warnings!
            assert any(
                all(
                    i in str(warning.message)
                    for i in [
                        "The POTCAR symbols for your bulk and defect calculations do not match",
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
                parse_projected_eigen=False,
            )
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
                parse_projected_eigen=False,
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
            for i, j in zip(cbm_orbital_diffs1, cbm_orbital_diffs2):
                print(f"cbm_orbital_diffs: {i:.3f} vs {j:.3f}")
                assert np.isclose(i, j, atol=orb_diff_tol)
            vbm_orbital_diffs1 = [subdict.pop("vbm_orbital_diff") for subdict in d1["states"]]
            vbm_orbital_diffs2 = [subdict.pop("vbm_orbital_diff") for subdict in d2["states"]]
            for i, j in zip(vbm_orbital_diffs1, vbm_orbital_diffs2):
                print(f"vbm_orbital_diffs: {i:.3f} vs {j:.3f}")
                assert np.isclose(i, j, atol=orb_diff_tol)

            orb_infos_orbitals1 = [
                subdict["vbm_info"]["orbital_info"].pop("orbitals") for subdict in d1["states"]
            ] + [subdict["cbm_info"]["orbital_info"].pop("orbitals") for subdict in d1["states"]]
            orb_infos_orbitals2 = [
                subdict["vbm_info"]["orbital_info"].pop("orbitals") for subdict in d2["states"]
            ] + [subdict["cbm_info"]["orbital_info"].pop("orbitals") for subdict in d2["states"]]
            for i, j in zip(orb_infos_orbitals1, orb_infos_orbitals2):
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
            f"{self.MgO_EXAMPLE_DIR}/Defects/Mg_O_+1/vasp_std",
            f"{self.MgO_EXAMPLE_DIR}/Defects/MgO_bulk/vasp_std",
            skip_corrections=True,
            parse_projected_eigen=True,
        ).defect_entry

        print("Testing MgO eigenvalue analysis")
        bes, fig = defect_entry.get_eigenvalue_analysis()  # Test plotting KS
        Mg_O_1_bes_path = f"{self.MgO_EXAMPLE_DIR}/Defects/Mg_O_1_band_edge_states.json"
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
        CdTe_defect_thermo = loadfn(os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_thermo_wout_meta.json"))
        v_Cd_minus1 = next(
            defect_entry
            for defect_entry in CdTe_defect_thermo.defect_entries
            if defect_entry.name == "v_Cd_-1"
        )
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
            "similar_orb_criterion=0.01, similar_energy_criterion=0.01 eV. Trying with the pydefect "
            "defaults of 0.2 and 0.5 eV." in str(w[0].message)
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
