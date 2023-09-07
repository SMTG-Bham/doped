"""
Tests for the `doped.plotting` module, which also implicitly tests most of the
`doped.utils.parsing` and `doped.analysis` modules.
"""

import os
import shutil
import unittest
from typing import Any, Dict

import matplotlib as mpl
import numpy as np
import pytest
from monty.serialization import loadfn
from test_vasp import _potcars_available

from doped import analysis, plotting
from doped.core import DefectEntry
from doped.utils.corrections import get_correction_freysoldt, get_correction_kumagai
from doped.utils.legacy_pmg.thermodynamics import DefectPhaseDiagram

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


# for pytest-mpl:
module_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(module_path, "data")


class CorrectionsPlottingTestCase(unittest.TestCase):
    module_path: str
    example_dir: str
    cdte_example_dir: str
    ytos_example_dir: str
    cdte_bulk_data_dir: str
    cdte_dielectric: np.ndarray
    v_Cd_dict: Dict[Any, Any]
    v_Cd_dpd: DefectPhaseDiagram
    F_O_1_entry: DefectEntry

    @classmethod
    def setUpClass(cls) -> None:
        # prepare parsed defect data (from doped parsing example)
        cls.module_path = os.path.dirname(os.path.abspath(__file__))
        cls.example_dir = os.path.join(cls.module_path, "../examples")
        cls.cdte_example_dir = os.path.join(cls.module_path, "../examples/CdTe")
        cls.ytos_example_dir = os.path.join(cls.module_path, "../examples/YTOS")
        cls.cdte_bulk_data_dir = os.path.join(cls.cdte_example_dir, "CdTe_bulk/vasp_ncl")
        cls.cdte_dielectric = np.array([[9.13, 0, 0], [0.0, 9.13, 0], [0, 0, 9.13]])  # CdTe

        cls.v_Cd_dict = {}  # dictionary of parsed vacancy defect entries

        for i in os.listdir(cls.cdte_example_dir):  # loops through the example directory
            if "v_Cd" in i:  # and parses folders that have "v_Cd" in their name
                print(f"Parsing {i}...")
                defect_path = f"{cls.cdte_example_dir}/{i}/vasp_ncl"
                cls.v_Cd_dict[i] = analysis.defect_entry_from_paths(
                    defect_path,
                    cls.cdte_bulk_data_dir,
                    cls.cdte_dielectric,
                    charge_state=None if _potcars_available() else int(i.split("_")[-1])  # to allow
                    # testing on GH Actions (otherwise test auto-charge determination if POTCARs available
                )

        cls.v_Cd_dpd = analysis.dpd_from_defect_dict(cls.v_Cd_dict)

        cls.F_O_1_entry = analysis.defect_entry_from_paths(
            defect_path=f"{cls.ytos_example_dir}/F_O_1",
            bulk_path=f"{cls.ytos_example_dir}/Bulk",
            dielectric=[40.7, 40.7, 25.2],
            charge_state=None if _potcars_available() else 1  # to allow testing on GH Actions
            # (otherwise test auto-charge determination if POTCARs available)
        )

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{data_dir}/remote_baseline_plots",
        filename="v_Cd_-2_FNV_plot.png",
        style=f"{module_path}/../doped/utils/doped.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_freysoldt(self):
        """
        Test FNV correction plotting.
        """
        return get_correction_freysoldt(self.v_Cd_dict["v_Cd_-2"], 9.13, plot=True, return_fig=True)

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{data_dir}/remote_baseline_plots",
        filename="F_O_+1_eFNV_plot.png",
        style=f"{module_path}/../doped/utils/doped.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_kumagai(self):
        """
        Test eFNV correction plotting.
        """
        return get_correction_kumagai(
            self.F_O_1_entry, dielectric=[40.7, 40.7, 25.2], plot=True, return_fig=True
        )


class DefectPlottingTestCase(unittest.TestCase):
    def test_format_defect_name(self):
        """
        Test _format_defect_name() function.
        """
        # test standard behaviour
        formatted_name = plotting._format_defect_name(
            defect_species="vac_1_Cd_0",
            include_site_info_in_name=False,
        )
        assert formatted_name == "$V_{Cd}^{0}$"
        # test with site number included
        formatted_name = plotting._format_defect_name(
            defect_species="vac_1_Cd_0",
            include_site_info_in_name=True,
        )
        assert formatted_name == "$V_{Cd_{1}}^{0}$"

        # test interstitial case with site number excluded
        formatted_name = plotting._format_defect_name(
            defect_species="Int_Cd_1_0",
            include_site_info_in_name=False,
        )
        assert formatted_name == "Cd$_i^{0}$"
        # test interstitial case with site number included
        formatted_name = plotting._format_defect_name(
            defect_species="Int_Cd_1_0",
            include_site_info_in_name=True,
        )
        assert formatted_name == "Cd$_{i_{1}}^{0}$"

        # test lowercase interstitial with site number excluded
        formatted_name = plotting._format_defect_name(
            defect_species="int_Cd_1_0",
            include_site_info_in_name=False,
        )
        assert formatted_name == "Cd$_i^{0}$"
        # test lowercase interstitial with site number included
        formatted_name = plotting._format_defect_name(
            defect_species="int_Cd_1_0",
            include_site_info_in_name=True,
        )
        assert formatted_name == "Cd$_{i_{1}}^{0}$"

        # test uppercase vacancy (pymatgen default name) with site number excluded
        formatted_name = plotting._format_defect_name(
            defect_species="Vac_1_Cd_0",
            include_site_info_in_name=False,
        )
        assert formatted_name == "$V_{Cd}^{0}$"
        # test uppercase vacancy (pymatgen default name) with site number included
        formatted_name = plotting._format_defect_name(
            defect_species="Vac_1_Cd_0",
            include_site_info_in_name=True,
        )
        assert formatted_name == "$V_{Cd_{1}}^{0}$"

        # test substitution with site number excluded
        formatted_name = plotting._format_defect_name(
            defect_species="as_1_Ni_on_Li_0",
            include_site_info_in_name=False,
        )
        assert formatted_name == "Ni$_{Li}^{0}$"

        # test substitution with site number included
        formatted_name = plotting._format_defect_name(
            defect_species="as_1_Ni_on_Li_0",
            include_site_info_in_name=True,
        )
        assert formatted_name == "Ni$_{Li_{1}}^{0}$"

        # test substitution with site number excluded, current doped format, two-letter subbed elt
        formatted_name = plotting._format_defect_name(
            defect_species="as_1_P_on_Na_-1",
            include_site_info_in_name=False,
        )
        assert formatted_name == "P$_{Na}^{-1}$"

        # test substitution with site number included, current doped format, two-letter subbed elt
        formatted_name = plotting._format_defect_name(
            defect_species="as_1_P_on_Na_-1 ",
            include_site_info_in_name=True,
        )
        assert formatted_name == "P$_{Na_{1}}^{-1}$"

        # test substitution with site number excluded, current doped format
        formatted_name = plotting._format_defect_name(
            defect_species="as_2_Na_on_P_0",
            include_site_info_in_name=False,
        )
        assert formatted_name == "Na$_{P}^{0}$"

        # test substitution with site number included, current doped format
        formatted_name = plotting._format_defect_name(
            defect_species="as_2_Na_on_P_0",
            include_site_info_in_name=True,
        )
        assert formatted_name == "Na$_{P_{2}}^{0}$"

        # test interstitial with site number excluded, current doped format
        formatted_name = plotting._format_defect_name(
            defect_species="inter_12_P_0",
            include_site_info_in_name=False,
        )
        assert formatted_name == "P$_i^{0}$"

        # test interstitial with site number included, current doped format
        formatted_name = plotting._format_defect_name(
            defect_species="inter_12_P_0",
            include_site_info_in_name=True,
        )
        assert formatted_name == "P$_{i_{12}}^{0}$"

        # test vacancy with site number excluded, current doped format
        formatted_name = plotting._format_defect_name(
            defect_species="vac_4_P_-2",
            include_site_info_in_name=False,
        )
        assert formatted_name == "$V_{P}^{-2}$"

        # test vacancy with site number included, current doped format
        formatted_name = plotting._format_defect_name(
            defect_species="vac_4_P_-2",
            include_site_info_in_name=True,
        )
        assert formatted_name == "$V_{P_{4}}^{-2}$"

        # check exceptions raised: invalid charge or defect_species
        # test error catching:
        with self.assertRaises(ValueError) as e:
            wrong_charge_error = ValueError(
                "Problem reading defect name vac_1_Cd_a, should end with charge state after "
                "underscore (e.g. vac_1_Cd_0)"
            )
            plotting._format_defect_name(defect_species="vac_1_Cd_a", include_site_info_in_name=True)
            assert wrong_charge_error in e.exception

        self.assertRaises(
            TypeError,
            plotting._format_defect_name,
            defect_species=2,
            include_site_info_in_name=True,
        )
        # check invalid defect type returns None
        assert (
            plotting._format_defect_name(defect_species="kk_Cd_1_0", include_site_info_in_name=True)
            is None
        )

        defect_species_name_dict = {
            "vac_Cd_mult32_0": "$V_{Cd}^{0}$",
            "VSb_0": "$V_{Sb}^{0}$",
            "VI_9": "$V_{I}^{+9}$",
            "V_Sb_0": "$V_{Sb}^{0}$",
            "V_I,_-2": "$V_{I}^{-2}$",
            "V_I_-2": "$V_{I}^{-2}$",
            "VacSb_2": "$V_{Sb}^{+2}$",
            "VacI_2": "$V_{I}^{+2}$",
            "Vac_Sb_3": "$V_{Sb}^{+3}$",
            "Vac_I_1": "$V_{I}^{+1}$",
            "VaSb_3": "$V_{Sb}^{+3}$",
            "VaI_9": "$V_{I}^{+9}$",
            "Va_Sb_10": "$V_{Sb}^{+10}$",
            "Va_I_4": "$V_{I}^{+4}$",
            "i_Sb_1": "Sb$_i^{+1}$",
            "Sb_i_3": "Sb$_i^{+3}$",
            "iSb_8": "Sb$_i^{+8}$",
            "IntSb_2": "Sb$_i^{+2}$",
            "Int_Sb_9": "Sb$_i^{+9}$",
            "Sb_Se_9": "Sb$_{Se}^{+9}$",
            "Sb_on_Se_9": "Sb$_{Se}^{+9}$",
            "Int_Li_mult64_-1": "Li$_i^{-1}$",
            "Int_Li_mult64_-2": "Li$_i^{-2}$",
            "Int_Li_mult64_0": "Li$_i^{0}$",
            "Int_Li_mult64_1": "Li$_i^{+1}$",
            "Int_Li_mult64_2": "Li$_i^{+2}$",
            "Sub_Li_on_Ni_mult32_-1": "Li$_{Ni}^{-1}$",
            "Sub_Li_on_Ni_mult32_-2": "Li$_{Ni}^{-2}$",
            "Sub_Li_on_Ni_mult32_0": "Li$_{Ni}^{0}$",
            "Sub_Li_on_Ni_mult32_+1": "Li$_{Ni}^{+1}$",
            "Sub_Li_on_Ni_mult32_2": "Li$_{Ni}^{+2}$",
            "Sub_Ni_on_Li_mult32_-1": "Ni$_{Li}^{-1}$",
            "Sub_Ni_on_Li_mult32_-2": "Ni$_{Li}^{-2}$",
            "Sub_Ni_on_Li_mult32_0": "Ni$_{Li}^{0}$",
            "Sub_Ni_on_Li_mult32_+1": "Ni$_{Li}^{+1}$",
            "Sub_Ni_on_Li_mult32_2": "Ni$_{Li}^{+2}$",
            "Vac_Li_mult32_-1": "$V_{Li}^{-1}$",
            "Vac_Li_mult32_-2": "$V_{Li}^{-2}$",
            "Vac_Li_mult32_0": "$V_{Li}^{0}$",
            "Vac_Li_mult32_+1": "$V_{Li}^{+1}$",
            "v_Cd_-1": "$V_{Cd}^{-1}$",
            "v_Te_+2": "$V_{Te}^{+2}$",
            "Cd_i_C3v_+2": "Cd$_i^{+2}$",
            "Cd_i_m32_2": "Cd$_i^{+2}$",
            "Cd_i_Td_Cd2.83_+2": "Cd$_i^{+2}$",
            "Cd_i_Td_Te2.83_2": "Cd$_i^{+2}$",
            "Te_i_C3vb_-2": "Te$_i^{-2}$",
            "Te_Cd_s32_2": "Te$_{Cd}^{+2}$",
            "Te_Cd_s32c_2": "Te$_{Cd}^{+2}$",
            "Cd_Te_+2": "Cd$_{Te}^{+2}$",
            "Cd_Te_2": "Cd$_{Te}^{+2}$",
            "as_2_Bi_on_O_-2": "Bi$_{O}^{-2}$",
            "S_Se_0": "S$_{Se}^{0}$",
            "Se_S_0": "Se$_{S}^{0}$",
            "Si_S_0": "Si$_{S}^{0}$",
            "S_Si_0": "S$_{Si}^{0}$",
        }

        for defect_species, expected_name in defect_species_name_dict.items():
            formatted_name = plotting._format_defect_name(
                defect_species=defect_species,
                include_site_info_in_name=False,
            )
            assert formatted_name == expected_name

        defect_species_w_site_info_name_dict = {
            "vac_Cd_mult32_0": "$V_{Cd_{m32}}^{0}$",
            "Int_Li_mult64_-1": "Li$_{i_{m64}}^{-1}$",
            "Int_Li_mult64_-2": "Li$_{i_{m64}}^{-2}$",
            "Int_Li_mult64_0": "Li$_{i_{m64}}^{0}$",
            "Int_Li_mult64_1": "Li$_{i_{m64}}^{+1}$",
            "Int_Li_mult64_2": "Li$_{i_{m64}}^{+2}$",
            "Sub_Li_on_Ni_mult32_-1": "Li$_{Ni_{m32}}^{-1}$",
            "Sub_Li_on_Ni_mult32_-2": "Li$_{Ni_{m32}}^{-2}$",
            "Sub_Li_on_Ni_mult32_0": "Li$_{Ni_{m32}}^{0}$",
            "Sub_Li_on_Ni_mult32_+1": "Li$_{Ni_{m32}}^{+1}$",
            "Sub_Li_on_Ni_mult32_2": "Li$_{Ni_{m32}}^{+2}$",
            "Sub_Ni_on_Li_mult32_-1": "Ni$_{Li_{m32}}^{-1}$",
            "Sub_Ni_on_Li_mult32_-2": "Ni$_{Li_{m32}}^{-2}$",
            "Sub_Ni_on_Li_mult32_0": "Ni$_{Li_{m32}}^{0}$",
            "Sub_Ni_on_Li_mult32_1": "Ni$_{Li_{m32}}^{+1}$",
            "Sub_Ni_on_Li_mult32_2": "Ni$_{Li_{m32}}^{+2}$",
            "Vac_Li_mult32_-1": "$V_{Li_{m32}}^{-1}$",
            "Vac_Li_mult32_-2": "$V_{Li_{m32}}^{-2}$",
            "Vac_Li_mult32_0": "$V_{Li_{m32}}^{0}$",
            "Vac_Li_mult32_1": "$V_{Li_{m32}}^{+1}$",
            "v_Cd_-1": "$V_{Cd}^{-1}$",
            "v_Te_2": "$V_{Te}^{+2}$",
            "v_Tea_2": "$V_{Te}^{+2}$",
            "Te_Cd_s32_2": "Te$_{Cd_{s32}}^{+2}$",
            "Te_Cd_s32c_2": "Te$_{Cd_{s32c}}^{+2}$",
            "Cd_Te_+2": "Cd$_{Te}^{+2}$",
            "Cd_Te_2": "Cd$_{Te}^{+2}$",
            "Cd_i_C3v_+2": "Cd$_{i_{C3v}}^{+2}$",
            "Cd_i_Td_Te2.83_2": "Cd$_{i_{Td-Te2.83}}^{+2}$",
            "Te_i_Td_Cd2.83_-2": "Te$_{i_{Td-Cd2.83}}^{-2}$",
            "v_O_C3v_0": "$V_{O_{C3v}}^{0}$",
            "v_O_C3v_1": "$V_{O_{C3v}}^{+1}$",
            "v_O_D4h_2": "$V_{O_{D4h}}^{+2}$",
            "Y_O_D4h_0": "Y$_{O_{D4h}}^{0}$",
            "Ti_Y_+2": "Ti$_{Y}^{+2}$",
            "Ti_O_D4h_0": "Ti$_{O_{D4h}}^{0}$",
            "Y_i_C4v_O1.92_0": "Y$_{i_{C4v-O1.92}}^{0}$",
            "Y_i_C4v_O2.68_1": "Y$_{i_{C4v-O2.68}}^{+1}$",
            "Y_i_Cs_O1.71_-2": "Y$_{i_{Cs-O1.71}}^{-2}$",
            "Y_i_Cs_O1.95_2": "Y$_{i_{Cs-O1.95}}^{+2}$",
            "Y_i_D2d_0": "Y$_{i_{D2d}}^{0}$",
            "v_O_C1_0": "$V_{O_{C1}}^{0}$",
            "v_O_C3_8": "$V_{O_{C3}}^{+8}$",
            "Li_i_C1_Li1.75_0": "Li$_{i_{C1-Li1.75}}^{0}$",
            "Li_i_C1_O1.72_1": "Li$_{i_{C1-O1.72}}^{+1}$",
            "Li_i_C1_O1.78_2": "Li$_{i_{C1-O1.78}}^{+2}$",
            "Li_i_C2_Li1.84O1.84_0": "Li$_{i_{C2-Li1.84O1.84}}^{0}$",
            "Li_i_C2_Li1.84O1.94_1": "Li$_{i_{C2-Li1.84O1.94}}^{+1}$",
            "Li_i_C2_Li1.86_2": "Li$_{i_{C2-Li1.86}}^{+2}$",
            "Li_i_C3_0": "Li$_{i_{C3}}^{0}$",
            "Cu_i_Oh_0": "Cu$_{i_{Oh}}^{0}$",
            "Cu_i_Td_0": "Cu$_{i_{Td}}^{0}$",
            "Cu_i_C3v_Ag1.56Cu1.56Ag2.99a_0": "Cu$_{i_{C3v-Ag1.56Cu1.56Ag2.99a}}^{0}$",
            "Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_0": "Cu$_{i_{C3v-Ag1.56Cu1.56Ag2.99b}}^{0}$",
            "Cu_i_C3v_Ag1.80_0": "Cu$_{i_{C3v-Ag1.80}}^{0}$",
            "Te_i_C1_Cd2.71Te2.71Cd4.25j_-2": "Te$_{i_{C1-Cd2.71Te2.71Cd4.25j}}^{-2}$",
            "Te_i_C1_Cd2.71Te2.71Cd4.25l_-2": "Te$_{i_{C1-Cd2.71Te2.71Cd4.25l}}^{-2}$",
            "v_Cd_C1_Te2.83Cd4.62Te5.42a_-1": "$V_{Cd_{C1-Te2.83Cd4.62Te5.42a}}^{-1}$",
            "v_Cd_C1_Te2.83Cd4.62Te5.42b_-1": "$V_{Cd_{C1-Te2.83Cd4.62Te5.42b}}^{-1}$",
            "v_Cd_C3v_Cd2.71_-1": "$V_{Cd_{C3v-Cd2.71}}^{-1}$",
            "v_Cd_C3v_Te2.83Cd4.25_-1": "$V_{Cd_{C3v-Te2.83Cd4.25}}^{-1}$",
            "v_Cd_C3v_Te2.83Cd4.62_-1": "$V_{Cd_{C3v-Te2.83Cd4.62}}^{-1}$",
        }
        for (
            defect_species,
            expected_name,
        ) in defect_species_w_site_info_name_dict.items():
            formatted_name = plotting._format_defect_name(
                defect_species=defect_species,
                include_site_info_in_name=True,
            )
            assert formatted_name == expected_name

    @pytest.mark.mpl_image_compare(
        baseline_dir=f"{data_dir}/remote_baseline_plots",
        filename="neutral_v_O_plot.png",
        style=f"{module_path}/../doped/utils/doped.mplstyle",
        savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
    )
    def test_plot_neutral_v_O_V2O5(self):
        """
        Test FNV correction plotting.
        """
        dielectric = [4.186, 19.33, 17.49]
        bulk_path = f"{data_dir}/V2O5/bulk"
        chempots = loadfn(f"{data_dir}/V2O5/chempots.json")

        defect_dict = {
            defect: analysis.defect_entry_from_paths(f"{data_dir}/V2O5/{defect}", bulk_path, dielectric)
            for defect in [dir for dir in os.listdir(f"{data_dir}/V2O5") if "v_O" in dir]
        }  # charge auto-determined (as neutral)
        dpd = analysis.dpd_from_defect_dict(defect_dict)
        return plotting.formation_energy_plot(dpd, chempots, facets=["VO2-V2O5"])
