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
import pandas as pd
import pytest
from monty.serialization import dumpfn, loadfn
from pymatgen.core.structure import Structure

from doped.analysis import (
    DefectParser,
    DefectsParser,
    defect_entry_from_paths,
    defect_from_structures,
    defect_name_from_structures,
    dpd_from_defect_dict,
    formation_energy_table,
)
from doped.generation import DefectsGenerator, get_defect_name_from_defect, get_defect_name_from_entry
from doped.plotting import formation_energy_plot
from doped.utils.parsing import (
    get_defect_site_idxs_and_unrelaxed_structure,
    get_defect_type_and_composition_diff,
    get_outcar,
    get_vasprun,
)

# for pytest-mpl:
module_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(module_path, "data")
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


# TODO: Add additional extrinsic defects test with our CdTe alkali defects (parsing & plotting)
# TODO: When adding full parsing tests (e.g. with full CdTe dpd or similar), add quick loop-over test of
#  the symmetry determination functions -> Use saved CdTe defect dict, and apply get_name function to
#  these defect entries


class DefectsParsingTestCase(unittest.TestCase):
    def setUp(self):
        self.module_path = os.path.dirname(os.path.abspath(__file__))
        self.EXAMPLE_DIR = os.path.join(self.module_path, "../examples")
        self.CdTe_EXAMPLE_DIR = os.path.join(self.module_path, "../examples/CdTe")
        self.CdTe_BULK_DATA_DIR = os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_bulk/vasp_ncl")
        self.CdTe_dielectric = np.array([[9.13, 0, 0], [0.0, 9.13, 0], [0, 0, 9.13]])  # CdTe

        self.YTOS_EXAMPLE_DIR = os.path.join(self.module_path, "../examples/YTOS")
        self.ytos_dielectric = [  # from legacy Materials Project
            [40.71948719643814, -9.282128210266565e-14, 1.26076160303219e-14],
            [-9.301652644020242e-14, 40.71948719776858, 4.149879443489052e-14],
            [5.311743673463141e-15, 2.041077680836527e-14, 25.237620491130023],
        ]

        self.Sb2Se3_DATA_DIR = os.path.join(self.module_path, "data/Sb2Se3")
        self.Sb2Se3_dielectric = np.array([[85.64, 0, 0], [0.0, 128.18, 0], [0, 0, 15.00]])

    def tearDown(self):
        if_present_rm(os.path.join(self.CdTe_BULK_DATA_DIR, "voronoi_nodes.json"))
        if_present_rm(os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_defect_dict.json"))
        if_present_rm(os.path.join(self.CdTe_EXAMPLE_DIR, "test_pop.json"))
        if_present_rm(os.path.join(self.YTOS_EXAMPLE_DIR, "Bulk", "voronoi_nodes.json"))
        if_present_rm(os.path.join(self.YTOS_EXAMPLE_DIR, "Y2Ti2S2O5_defect_dict.json"))

    def _check_DefectsParser(self, dp, skip_corrections=False):
        # check generating dpd and plot:
        dpd = dpd_from_defect_dict(dp.defect_dict)
        with warnings.catch_warnings(record=True) as w:
            formation_energy_plot(dpd)
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

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{data_dir}/remote_baseline_plots",
        filename="CdTe_example_defects_plot.png",
        style=f"{module_path}/../doped/utils/doped.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_DefectsParser_CdTe(self):
        with warnings.catch_warnings(record=True) as w:
            CdTe_dp = DefectsParser(output_path=self.CdTe_EXAMPLE_DIR, dielectric=9.13)
        print([warn.message for warn in w])  # for debugging
        assert not any(
            "The KPOINTS for your bulk and defect calculations do not match" in str(warn.message)
            for warn in w
        )  # no KPOINTS warning
        assert any(
            all(
                i in str(warn.message)
                for i in [
                    f"Warning(s) encountered when parsing Int_Te_3_Unperturbed_1 at "
                    f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_Unperturbed_1/vasp_ncl:",
                    "There are mismatching INCAR tags for your bulk and defect calculations",
                ]
            )
            for warn in w
        )  # INCAR warning
        assert any(
            all(
                i in str(warn.message)
                for i in [
                    f"Warning(s) encountered when parsing Int_Te_3_2 at "
                    f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl",
                    "Multiple `OUTCAR` files found in defect directory",
                ]
            )
            for warn in w
        )  # multiple OUTCARs warning
        assert any(
            "Beware: The Freysoldt (FNV) charge correction scheme has been used for some "
            "defects, while the Kumagai (eFNV) scheme has been used for others." in str(warn.message)
            for warn in w
        )  # multiple corrections warning

        CdTe_dpd = dpd_from_defect_dict(CdTe_dp.defect_dict)
        dumpfn(CdTe_dpd, os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_example_dpd.json"))  # for test_plotting
        with warnings.catch_warnings(record=True) as w:
            formation_energy_plot(CdTe_dpd)
        print([warn.message for warn in w])  # for debugging
        assert any("You have not specified chemical potentials" in str(warn.message) for warn in w)
        assert any(
            "All formation energies for Int_Te_3_2 are below zero" in str(warn.message) for warn in w
        )
        assert any("All formation energies for Int_Te_3_Unperturbed_1" in str(warn.message) for warn in w)

        # test attributes:
        assert CdTe_dp.output_path == self.CdTe_EXAMPLE_DIR
        assert CdTe_dp.dielectric == 9.13
        assert CdTe_dp.error_tolerance == 0.05
        assert CdTe_dp.bulk_path == self.CdTe_BULK_DATA_DIR  # automatically determined
        assert CdTe_dp.subfolder == "vasp_ncl"  # automatically determined
        assert CdTe_dp.bulk_bandgap_path is None

        self._check_DefectsParser(CdTe_dp)
        assert os.path.exists(os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_defect_dict.json"))
        if_present_rm(os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_defect_dict.json"))

        # explicitly check some formation energies
        def _check_parsed_CdTe_defect_energies(dp):
            assert np.isclose(
                dp.defect_dict["v_Cd_0"].get_ediff() - sum(dp.defect_dict["v_Cd_0"].corrections.values()),
                4.166,
                atol=3e-3,
            )  # uncorrected energy
            assert np.isclose(dp.defect_dict["v_Cd_0"].get_ediff(), 4.166, atol=1e-3)
            assert np.isclose(dp.defect_dict["v_Cd_-1"].get_ediff(), 6.355, atol=1e-3)
            assert np.isclose(
                dp.defect_dict["v_Cd_-2"].get_ediff()
                - sum(dp.defect_dict["v_Cd_-2"].corrections.values()),
                7.661,
                atol=3e-3,
            )  # uncorrected energy
            assert np.isclose(dp.defect_dict["v_Cd_-2"].get_ediff(), 8.398, atol=1e-3)
            assert np.isclose(dp.defect_dict["Int_Te_3_2"].get_ediff(), -6.2009, atol=1e-3)

        _check_parsed_CdTe_defect_energies(CdTe_dp)

        assert len(CdTe_dp.defect_folders) == 7
        for name in CdTe_dp.defect_dict:
            assert name in CdTe_dp.defect_folders  # all folder names recognised for CdTe examples

        # both OUTCARs and LOCPOTs in CdTe folders
        assert len(CdTe_dp.bulk_corrections_data) == 2
        for _k, v in CdTe_dp.bulk_corrections_data.items():
            assert v is not None

        # test no dielectric and no JSON:
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(output_path=self.CdTe_EXAMPLE_DIR, json_filename=False)
        print([warn.message for warn in w])  # for debugging
        assert any(
            "The dielectric constant (`dielectric`) is needed to compute finite-size charge "
            "corrections, but none was provided" in str(warn.message)
            for warn in w
        )
        self._check_DefectsParser(dp, skip_corrections=True)
        assert not os.path.exists(os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_defect_dict.json"))

        # test custom settings:
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(
                output_path=self.CdTe_EXAMPLE_DIR,
                dielectric=[9.13, 9.13, 9.13],
                error_tolerance=0.01,
                skip_corrections=False,
                bulk_bandgap_path=self.CdTe_BULK_DATA_DIR,
                processes=4,
                json_filename="test_pop.json",
            )
        print([warn.message for warn in w])  # for debugging
        assert any(
            "Estimated error in the Kumagai (eFNV) charge correction for defect Int_Te_3_2 is "
            "0.012 eV (i.e. which is greater than the `error_tolerance`: 0.010 eV)" in str(warn.message)
            for warn in w
        )
        assert os.path.exists(os.path.join(self.CdTe_EXAMPLE_DIR, "test_pop.json"))

        self._check_DefectsParser(dp)
        _check_parsed_CdTe_defect_energies(dp)  # same energies as above

        # test changed attributes:
        assert dp.output_path == self.CdTe_EXAMPLE_DIR
        assert dp.dielectric == [9.13, 9.13, 9.13]
        assert dp.error_tolerance == 0.01
        assert dp.bulk_bandgap_path == self.CdTe_BULK_DATA_DIR
        assert dp.processes == 4
        self._check_DefectsParser(dp)

        # test setting subfolder to unrecognised one:
        with self.assertRaises(FileNotFoundError) as exc:
            dp = DefectsParser(output_path=self.CdTe_EXAMPLE_DIR, subfolder="vasp_gam")
        assert (
            f"`vasprun.xml(.gz)` files (needed for defect parsing) not found in bulk folder at: "
            f"{self.CdTe_EXAMPLE_DIR}/CdTe_bulk or subfolder: vasp_gam - please ensure `vasprun.xml(.gz)` "
            f"files are present and/or specify `bulk_path` manually."
        ) in str(exc.exception)

        dp = DefectsParser(output_path=self.CdTe_EXAMPLE_DIR, processes=1, dielectric=9.13)
        reloaded_defect_dict = loadfn(os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_defect_dict.json"))

        for defect_dict in [dp.defect_dict, reloaded_defect_dict]:
            for defect_name, defect_entry in defect_dict.items():
                assert defect_entry.name == CdTe_dp.defect_dict[defect_name].name
                assert np.isclose(defect_entry.get_ediff(), CdTe_dp.defect_dict[defect_name].get_ediff())
                assert np.allclose(
                    defect_entry.sc_defect_frac_coords,
                    CdTe_dp.defect_dict[defect_name].sc_defect_frac_coords,
                )

        # skip_corrections:
        dp = DefectsParser(output_path=self.CdTe_EXAMPLE_DIR, skip_corrections=True)
        self._check_DefectsParser(dp, skip_corrections=True)

        # anisotropic dielectric
        fake_aniso_dielectric = [1, 2, 3]
        with warnings.catch_warnings(record=True) as w:
            dp = DefectsParser(output_path=self.CdTe_EXAMPLE_DIR, dielectric=fake_aniso_dielectric)
        print([warn.message for warn in w])  # for debugging
        assert any(
            "Estimated error in the Kumagai (eFNV) charge correction for defect Int_Te_3_2 is "
            "0.157 eV (i.e. which is greater than the `error_tolerance`: 0.050 eV)." in str(warn.message)
            for warn in w
        )
        assert any(
            all(
                i in str(warn.message)
                for i in [
                    f"An anisotropic dielectric constant was supplied, but `OUTCAR` files (needed to "
                    f"compute the _anisotropic_ Kumagai eFNV charge correction) were not found in the "
                    f"defect (at {self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl) & bulk",
                    "`LOCPOT` files were found in both defect & bulk folders, and so the Freysoldt (FNV) "
                    "charge correction developed for _isotropic_ materials will be applied here, "
                    "which corresponds to using the effective isotropic average of the supplied "
                    "anisotropic dielectric. This could lead to significant errors for very anisotropic "
                    "systems and/or relatively small supercells!",
                ]
            )
            for warn in w
        )
        self._check_DefectsParser(dp)

        # integration test using parsed CdTe dpd and chempots for plotting:
        CdTe_chempots = loadfn(os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_chempots.json"))

        return formation_energy_plot(CdTe_dpd, chempots=CdTe_chempots, facets=["CdTe-Te"])

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{data_dir}/remote_baseline_plots",
        filename="YTOS_example_defects_plot.png",
        style=f"{module_path}/../doped/utils/doped.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_DefectsParser_YTOS(self):
        # bulk path needs to be specified for YTOS as it's not the default name:
        dp = DefectsParser(
            output_path=self.YTOS_EXAMPLE_DIR,
            bulk_path=os.path.join(self.YTOS_EXAMPLE_DIR, "Bulk"),
            dielectric=self.ytos_dielectric,
        )
        self._check_DefectsParser(dp)
        dpd = dpd_from_defect_dict(dp.defect_dict)
        dumpfn(dpd, os.path.join(self.YTOS_EXAMPLE_DIR, "YTOS_example_dpd.json"))  # for test_plotting
        return formation_energy_plot(dpd)  # no chempots for YTOS formation energy plot test

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{data_dir}/remote_baseline_plots",
        filename="O_Se_example_defects_plot.png",
        style=f"{module_path}/../doped/utils/doped.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_extrinsic_Sb2Se3(self):
        with self.assertRaises(ValueError) as exc:
            DefectsParser(
                output_path=f"{self.Sb2Se3_DATA_DIR}/defect",
                dielectric=self.Sb2Se3_dielectric,
            )
        assert (  # bulk in separate folder so fails
            f"Could not automatically determine bulk supercell calculation folder in "
            f"{self.Sb2Se3_DATA_DIR}/defect, found 0 folders containing `vasprun.xml(.gz)` files (in "
            f"subfolders) and 'bulk' in the folder name" in str(exc.exception)
        )

        with warnings.catch_warnings(record=True) as w:  # no warning about negative corrections with
            # strong anisotropic dielectric:
            Sb2Se3_O_dp = DefectsParser(
                output_path=f"{self.Sb2Se3_DATA_DIR}/defect",
                bulk_path=f"{self.Sb2Se3_DATA_DIR}/bulk",
                dielectric=self.Sb2Se3_dielectric,
            )
        print([warn.message for warn in w])  # for debugging
        assert not w  # no warnings
        Sb2Se3_O_dpd = dpd_from_defect_dict(Sb2Se3_O_dp.defect_dict)
        dumpfn(Sb2Se3_O_dpd, os.path.join(self.Sb2Se3_DATA_DIR, "Sb2Se3_O_example_dpd.json"))  # for
        # test_plotting

        # warning about negative corrections when using (fake) isotropic dielectric:
        with warnings.catch_warnings(record=True) as w:
            Sb2Se3_O_dp = DefectsParser(
                output_path=f"{self.Sb2Se3_DATA_DIR}/defect",
                bulk_path=f"{self.Sb2Se3_DATA_DIR}/bulk",
                dielectric=40,  # fake isotropic dielectric
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

        return formation_energy_plot(Sb2Se3_O_dpd, chempots={"O": -8.9052, "Se": -5})  # example chempots


class DopedParsingTestCase(unittest.TestCase):
    def setUp(self):
        self.module_path = os.path.dirname(os.path.abspath(__file__))
        self.EXAMPLE_DIR = os.path.join(self.module_path, "../examples")
        self.CdTe_EXAMPLE_DIR = os.path.join(self.module_path, "../examples/CdTe")
        self.YTOS_EXAMPLE_DIR = os.path.join(self.module_path, "../examples/YTOS")
        self.CdTe_BULK_DATA_DIR = os.path.join(self.CdTe_EXAMPLE_DIR, "CdTe_bulk/vasp_ncl")
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

        parsed_v_cd_m2 = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=self.CdTe_BULK_DATA_DIR,
            dielectric=self.CdTe_dielectric,
        )

        parsed_v_cd_m2_explicit_charge = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=self.CdTe_BULK_DATA_DIR,
            dielectric=self.CdTe_dielectric,
            charge_state=-2,
        )
        assert parsed_v_cd_m2.get_ediff() == parsed_v_cd_m2_explicit_charge.get_ediff()
        assert parsed_v_cd_m2.charge_state == -2
        assert parsed_v_cd_m2_explicit_charge.charge_state == -2

        # Check that the correct Freysoldt correction is applied
        correct_correction_dict = {
            "freysoldt_charge_correction": 0.7376460317828045,
        }
        for correction_name, correction_energy in correct_correction_dict.items():
            for defect_entry in [parsed_v_cd_m2, parsed_v_cd_m2_explicit_charge]:
                assert np.isclose(
                    defect_entry.corrections[correction_name],
                    correction_energy,
                    atol=1e-3,
                )

        # test warning when specifying the wrong charge:
        with warnings.catch_warnings(record=True) as w:
            parsed_v_cd_m1 = defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=self.CdTe_dielectric,
                charge_state=-1,
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
        ytos_F_O_1 = defect_entry_from_paths(
            f"{self.YTOS_EXAMPLE_DIR}/F_O_1",
            f"{self.YTOS_EXAMPLE_DIR}/Bulk",
            self.ytos_dielectric,
            skip_corrections=True,
        )
        assert ytos_F_O_1.charge_state == 1
        assert np.isclose(ytos_F_O_1.get_ediff(), -0.0852, atol=1e-3)  # uncorrected energy

        ytos_F_O_1 = defect_entry_from_paths(  # with corrections this time
            f"{self.YTOS_EXAMPLE_DIR}/F_O_1",
            f"{self.YTOS_EXAMPLE_DIR}/Bulk",
            self.ytos_dielectric,
        )
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
            parsed_v_cd_m2_fake_aniso = DefectParser.from_paths(
                defect_path=defect_path,
                bulk_path=self.CdTe_BULK_DATA_DIR,
                dielectric=fake_aniso_dielectric,
            ).defect_entry
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert (
                f"An anisotropic dielectric constant was supplied, but `OUTCAR` files (needed to compute "
                f"the _anisotropic_ Kumagai eFNV charge correction) were not found in the defect "
                f"(at {defect_path}) & bulk (at {self.CdTe_BULK_DATA_DIR}) folders.\n`LOCPOT` files were "
                f"found in both defect & bulk folders, and so the Freysoldt (FNV) charge correction "
                f"developed for _isotropic_ materials will be applied here, which corresponds to using "
                f"the effective isotropic average of the supplied anisotropic dielectric. This could "
                f"lead to significant errors for very anisotropic systems and/or relatively small "
                f"supercells!" in str(w[-1].message)
            )

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
            ).defect_entry
            assert (
                f"Multiple `OUTCAR` files found in defect directory:"
                f" {self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl. Using"
                f" {self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/OUTCAR.gz to parse core levels and "
                f"compute the Kumagai (eFNV) image charge correction." in str(w[0].message)
            )
            assert (
                f"Estimated error in the Kumagai (eFNV) charge correction for defect "
                f"{parsed_int_Te_2_fake_aniso.name} is 0.157 eV (i.e. which is greater than the "
                f"`error_tolerance`: 0.050 eV). You may want to check the accuracy of the correction "
                f"by plotting the site potential differences (using "
                f"`defect_entry.get_kumagai_correction()` with `plot=True`). Large errors are often due "
                f"to unstable or shallow defect charge states (which can't be accurately modelled with "
                f"the supercell approach). If this error is not acceptable, you may need to use a larger "
                f"supercell for more accurate energies." in str(w[1].message)
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
                f"`LOCPOT` or `OUTCAR` files are not present in both the defect (at {defect_path}) and "
                f"bulk (at {self.CdTe_BULK_DATA_DIR}) folders. These are needed to perform the "
                f"finite-size charge corrections. Charge corrections will not be applied for this defect."
                in str(w[0].message)
            )

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
            )
            assert len(w) == 0

        assert np.isclose(
            parsed_v_cd_0.get_ediff() - sum(parsed_v_cd_0.corrections.values()), 4.166, atol=3e-3
        )  # uncorrected energy
        assert np.isclose(parsed_v_cd_0.get_ediff(), 4.166, atol=1e-3)

    def _check_no_icorelevel_warning_int_te(self, dielectric, warnings, num_warnings, action):
        result = defect_entry_from_paths(
            defect_path=f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl",
            bulk_path=self.CdTe_BULK_DATA_DIR,
            dielectric=dielectric,
            charge_state=2,
        )
        assert len(warnings) == num_warnings
        assert all(issubclass(warning.category, UserWarning) for warning in warnings)
        assert (  # different warning start depending on whether isotropic or anisotropic dielectric
            f"in the defect (at {self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl) & bulk (at "
            f"{self.CdTe_BULK_DATA_DIR}) folders were unable to be parsed, giving the following error "
            f"message:\nUnable to parse atomic core potentials from defect `OUTCAR` at "
            f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/OUTCAR_no_core_levels.gz. This can happen if "
            f"`ICORELEVEL` was not set to 0 (= default) in the `INCAR`, or if the calculation was "
            f"finished prematurely with a `STOPCAR`. The Kumagai charge correction cannot be computed "
            f"without this data!\n{action}" in str(warnings[0].message)
        )

        return result

    def _parse_Int_Te_3_2_and_count_warnings(self, fake_aniso_dielectric, w, num_warnings):
        defect_entry_from_paths(
            defect_path=f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl",
            bulk_path=self.CdTe_BULK_DATA_DIR,
            dielectric=fake_aniso_dielectric,
            charge_state=2,
        )
        print([warn.message for warn in w])  # for debugging
        assert len(w) == num_warnings
        # defect and bulk)
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
                f"Multiple `OUTCAR` files found in bulk directory: {self.CdTe_BULK_DATA_DIR}. Using"
                f" {self.CdTe_BULK_DATA_DIR}/OUTCAR.gz to parse core levels and compute the Kumagai ("
                f"eFNV) image charge correction." in str(w[0].message)
            )
            assert (
                f"Multiple `OUTCAR` files found in defect directory:"
                f" {self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl. Using"
                f" {self.CdTe_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/OUTCAR.gz to parse core levels and "
                f"compute the Kumagai (eFNV) image charge correction." in str(w[1].message)
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
            )
            assert len(w) == 2  # multiple LOCPOTs (both defect and bulk)
            assert all(issubclass(warning.category, UserWarning) for warning in w)
            assert (
                f"Multiple `LOCPOT` files found in bulk directory: {self.CdTe_BULK_DATA_DIR}. Using"
                f" {self.CdTe_BULK_DATA_DIR}/LOCPOT.gz to parse the electrostatic potential and compute "
                f"the Freysoldt (FNV) charge correction." in str(w[0].message)
            )
            assert (
                f"Multiple `LOCPOT` files found in defect directory: {defect_path}. Using"
                f" {defect_path}/LOCPOT.gz to parse the electrostatic potential and compute the "
                f"Freysoldt (FNV) charge correction." in str(w[1].message)
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
            )
            assert len(w) == 2  # multiple `vasprun.xml`s (both defect and bulk)
            assert all(issubclass(warning.category, UserWarning) for warning in w)
            assert (
                f"Multiple `vasprun.xml` files found in bulk directory: {self.CdTe_BULK_DATA_DIR}. Using"
                f" {self.CdTe_BULK_DATA_DIR}/vasprun.xml.gz to parse the calculation energy and metadata."
                in str(w[0].message)
            )
            assert (
                f"Multiple `vasprun.xml` files found in defect directory: {defect_path}. Using"
                f" {defect_path}/vasprun.xml.gz to parse the calculation energy and metadata."
                in str(w[1].message)
            )

    def test_dielectric_initialisation(self):
        """
        Test that dielectric can be supplied as float or int or 3x1 array/list
        or 3x3 array/list.
        """
        defect_path = f"{self.CdTe_EXAMPLE_DIR}/v_Cd_-2/vasp_ncl"
        # get correct Freysoldt correction energy:
        parsed_v_cd_m2 = defect_entry_from_paths(  # defect charge determined automatically
            defect_path=defect_path,
            bulk_path=self.CdTe_BULK_DATA_DIR,
            dielectric=self.CdTe_dielectric,
            charge_state=-2,
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
            if "v_Cd" in i:  # loop folders and parse those with "v_Cd" in name
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

    def test_extrinsic_interstitial_parsing_and_kumagai(self):
        """
        Test parsing of extrinsic F in YTOS interstitial and Kumagai-Oba (eFNV)
        correction.
        """
        defect_path = f"{self.YTOS_EXAMPLE_DIR}/Int_F_-1/"

        # parse with no transformation.json or explicitly-set-charge:
        with warnings.catch_warnings(record=True) as w:
            int_F_minus1_ent = defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk/",
                dielectric=self.ytos_dielectric,
            )
        assert not [warning for warning in w if issubclass(warning.category, UserWarning)]

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
            f"{int_F_minus1_ent.name} is 0.003 eV (i.e. which is greater than the `error_tolerance`: "
            f"0.001 eV). You may want to check the accuracy of the correction by plotting the site "
            f"potential differences (using `defect_entry.get_kumagai_correction()` with "
            f"`plot=True`). Large errors are often due to unstable or shallow defect charge states ("
            f"which can't be accurately modelled with the supercell approach). If this error is not "
            f"acceptable, you may need to use a larger supercell for more accurate energies."
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

        # test returning correction error with plot:
        corr, fig, corr_error = int_F_minus1_ent.get_kumagai_correction(
            return_correction_error=True, plot=True
        )
        assert np.isclose(corr.correction_energy, correction_dict["kumagai_charge_correction"], atol=1e-3)
        assert np.isclose(corr_error, 0.003, atol=1e-3)

        # test just correction returned with plot = False and return_correction_error = False:
        corr = int_F_minus1_ent.get_kumagai_correction()
        assert np.isclose(corr.correction_energy, correction_dict["kumagai_charge_correction"], atol=1e-3)

        # test symmetry determination (warning here because periodicity breaking affects F_i):
        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            relaxed_defect_name = get_defect_name_from_entry(int_F_minus1_ent)
            assert len(w) == 1
            assert (
                "`unrelaxed` is set to False (i.e. get _relaxed_ defect symmetry), but doped has "
                "detected that the supercell is a non-scalar matrix expansion which is breaking "
                "the cell periodicity, likely preventing the correct point group symmetry "
                in str(w[-1].message)
            )
        assert relaxed_defect_name == "F_i_C4v_O2.67"
        assert get_defect_name_from_entry(int_F_minus1_ent, unrelaxed=True) == "F_i_Cs_O2.67"

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
            )  # check no correction error warning with default tolerance:

        assert any(
            f"Estimated error in the Freysoldt (FNV) charge correction for defect {F_O_1_ent.name} is "
            f"0.000 eV (i.e. which is greater than the `error_tolerance`: 0.000 eV). You may want to "
            f"check the accuracy of the correction by plotting the site potential differences (using "
            f"`defect_entry.get_freysoldt_correction()` with `plot=True`). Large errors are often due to "
            f"unstable or shallow defect charge states (which can't be accurately modelled with the "
            f"supercell approach). If this error is not acceptable, you may need to use a larger "
            f"supercell for more accurate energies." in str(warning.message)
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

        # test returning correction error with plot:
        corr, fig, corr_error = F_O_1_ent.get_freysoldt_correction(return_correction_error=True, plot=True)
        assert np.isclose(corr.correction_energy, 0.11670254204631794, atol=1e-3)
        assert np.isclose(corr_error, 0.000, atol=1e-3)

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
        )

        self._test_F_O_1_ent(F_O_1_ent, 0.04176, "kumagai_charge_correction", 0.12699488572686776)

        # test symmetry determination (no warning here because periodicity breaking doesn't affect F_O):
        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            relaxed_defect_name = get_defect_name_from_entry(F_O_1_ent)
            assert len(w) == 0
        assert relaxed_defect_name == "F_O_D4h_Ti1.79"
        assert get_defect_name_from_entry(F_O_1_ent, unrelaxed=True) == "F_O_D4h_Ti1.79"

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
        interstitial sites (Te_i^+1 ground-state and metastable from Kavanagh
        et al.

        2022 doi.org/10.1039/D2FD00043A).
        """
        from pydefect.analyzer.calc_results import CalcResults
        from pydefect.cli.vasp.make_efnv_correction import make_efnv_correction

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


class DopedParsingFunctionsTestCase(unittest.TestCase):
    def setUp(self):
        DopedParsingTestCase.setUp(self)  # get attributes from DopedParsingTestCase
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.prim_cdte = Structure.from_file(f"{self.EXAMPLE_DIR}/CdTe/relaxed_primitive_POSCAR")
        self.ytos_bulk_supercell = Structure.from_file(f"{self.EXAMPLE_DIR}/YTOS/Bulk/POSCAR")
        self.lmno_primitive = Structure.from_file(f"{self.data_dir}/Li2Mn3NiO8_POSCAR")
        self.non_diagonal_ZnS = Structure.from_file(f"{self.data_dir}/non_diagonal_ZnS_supercell_POSCAR")

        # TODO: Try rattling the structures (and modifying symprec a little to test tolerance?)

    def tearDown(self):
        if_present_rm(os.path.join(self.CdTe_BULK_DATA_DIR, "voronoi_nodes.json"))
        if_present_rm(os.path.join(self.YTOS_EXAMPLE_DIR, "Bulk", "voronoi_nodes.json"))
        if_present_rm("./vasprun.xml")

    def test_defect_name_from_structures(self):
        # by proxy also tests defect_from_structures
        for struct in [
            self.prim_cdte,
            self.ytos_bulk_supercell,
            self.lmno_primitive,
            self.non_diagonal_ZnS,
        ]:
            defect_gen = DefectsGenerator(struct)
            for defect_entry in [entry for entry in defect_gen.values() if entry.charge_state == 0]:
                print(
                    defect_from_structures(defect_entry.bulk_supercell, defect_entry.defect_supercell),
                    defect_entry.defect_supercell_site,
                )
                assert defect_name_from_structures(
                    defect_entry.bulk_supercell, defect_entry.defect_supercell
                ) == get_defect_name_from_defect(defect_entry.defect)

                # Can't use defect.structure/defect.defect_structure because might be vacancy in a 1/2
                # atom cell etc.:
                # assert defect_name_from_structures(
                #     defect_entry.defect.structure, defect_entry.defect.defect_structure
                # ) == get_defect_name_from_defect(defect_entry.defect)

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
        with gzip.open(
            f"{self.CdTe_EXAMPLE_DIR}/Int_Te_3_Unperturbed_1/vasp_ncl/vasprun.xml.gz", "rb"
        ) as f_in, open("./vasprun.xml", "wb") as f_out:
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
            )
        assert any("Detected atoms far from the defect site" in str(warning.message) for warning in w)


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


class AnalysisFunctionsTestCase(unittest.TestCase):
    """
    Test post-processing analysis functions.
    """

    def setUp(self):
        self.module_path = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.sb2o5_chempots = loadfn(f"{self.data_dir}/Sb2O5/Sb2O5_chempots.json")
        self.sb2o5_dpd = loadfn(f"{self.data_dir}/Sb2O5/sb2o5_dpd.json")

    def tearDown(self):
        if_present_rm("test.csv")

    def test_formation_energy_table(self):
        def _check_formation_energy_table(formation_energy_table_df, fermi_level=0, dpd=self.sb2o5_dpd):
            defect_entry_names = [defect_entry.name for defect_entry in dpd.entries]
            assert sorted(formation_energy_table_df["Defect"].tolist()) == sorted(defect_entry_names)

            # for each row, assert sum of formation energy terms equals formation energy column
            np.isclose(
                np.asarray(sum(formation_energy_table_df.iloc[:, i] for i in range(2, 8))),
                np.asarray(formation_energy_table_df.iloc[:, 8]),
                atol=2e-3,
            )

            assert np.isclose(
                formation_energy_table_df.iloc[:, 1] * fermi_level,
                formation_energy_table_df.iloc[:, 4],
                atol=2e-3,
            ).all()

        formation_energy_table_df = formation_energy_table(
            self.sb2o5_dpd, self.sb2o5_chempots, facets=["Sb2O5-SbO2"], fermi_level=3
        )
        _check_formation_energy_table(formation_energy_table_df, fermi_level=3)

        formation_energy_table_df = formation_energy_table(  # test default with E_F = 0
            self.sb2o5_dpd,
            self.sb2o5_chempots,
            facets=["Sb2O5-O2"],
        )
        _check_formation_energy_table(formation_energy_table_df, fermi_level=0)

        formation_energy_table_df_manual_chempots = formation_energy_table(  # test default with E_F = 0
            self.sb2o5_dpd,
            chempots=self.sb2o5_chempots["facets_wrt_el_refs"]["Sb2O5-O2"],
            el_refs=self.sb2o5_chempots["elemental_refs"],
        )
        _check_formation_energy_table(formation_energy_table_df_manual_chempots, fermi_level=0)

        # check manual and auto chempots the same:
        assert formation_energy_table_df_manual_chempots.equals(formation_energy_table_df)

        # assert runs fine without chempots:
        formation_energy_table_df = formation_energy_table(self.sb2o5_dpd)
        _check_formation_energy_table(formation_energy_table_df)

        # assert runs fine with only raw chempots:
        formation_energy_table_df = formation_energy_table(
            self.sb2o5_dpd, chempots=self.sb2o5_chempots["facets"]["Sb2O5-O2"]
        )
        _check_formation_energy_table(formation_energy_table_df)
        # check same formation energies as with manual chempots plus el_refs:
        assert formation_energy_table_df.iloc[:, 8].equals(
            formation_energy_table_df_manual_chempots.iloc[:, 8]
        )

        # check saving to csv and reloading all works fine:
        formation_energy_table_df.to_csv("test.csv", index=False)
        formation_energy_table_df_reloaded = pd.read_csv("test.csv")
        assert formation_energy_table_df_reloaded.equals(formation_energy_table_df)
