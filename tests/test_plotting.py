"""
Tests for the `doped.utils.plotting` module, which also implicitly tests most
of the `doped.utils.parsing` and `doped.analysis` modules.

Note that some of the integration tests in `test_analysis.py` also implicitly
tests much of the plotting functionality.
"""

import os
import shutil
import unittest
import warnings
from copy import deepcopy

import matplotlib as mpl
import pytest
from monty.serialization import loadfn
from test_thermodynamics import DefectThermodynamicsSetupMixin, custom_mpl_image_compare, data_dir

from doped import analysis
from doped.thermodynamics import DefectThermodynamics
from doped.utils import plotting

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


class DefectPlottingTestCase(unittest.TestCase):
    def setUp(self):
        self.module_path = os.path.dirname(os.path.abspath(__file__))
        self.EXAMPLE_DIR = os.path.join(self.module_path, "../examples")
        self.CdTe_EXAMPLE_DIR = os.path.join(self.module_path, "../examples/CdTe")
        self.CdTe_thermo = loadfn(f"{self.CdTe_EXAMPLE_DIR}/CdTe_example_thermo.json")
        self.CdTe_chempots = loadfn(f"{self.CdTe_EXAMPLE_DIR}/CdTe_chempots.json")
        self.YTOS_EXAMPLE_DIR = os.path.join(self.module_path, "../examples/YTOS")
        self.YTOS_thermo = loadfn(f"{self.YTOS_EXAMPLE_DIR}/YTOS_example_thermo.json")

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_plot_CdTe(self):
        return self.CdTe_thermo.plot(self.CdTe_chempots, limit="CdTe-Te")

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot_listed_colormap.png")
    def test_plot_CdTe_custom_listed_colormap(self):
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(["#D4447E", "#E9A66C", "#507BAA", "#5FABA2", "#63666A"])
        return self.CdTe_thermo.plot(self.CdTe_chempots, colormap=cmap)[1]

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot_viridis.png")
    def test_plot_CdTe_custom_str_colormap(self):
        return self.CdTe_thermo.plot(self.CdTe_chempots, colormap="viridis")[1]

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_plot_CdTe_multiple_figs(self):
        # when limits not specified, plots all of them (second should be Te-rich here)
        return self.CdTe_thermo.plot(self.CdTe_chempots)[1]

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot_Cd_rich.png")
    def test_plot_CdTe_Cd_rich(self):
        # when limits not specified, plots all of them (first should be Cd-rich here)
        return self.CdTe_thermo.plot(self.CdTe_chempots)[0]

    @custom_mpl_image_compare(filename="YTOS_example_defects_plot.png")
    def test_plot_YTOS(self):
        with warnings.catch_warnings(record=True) as w:
            plot = self.YTOS_thermo.plot()  # no chempots
        print([str(warn.message) for warn in w])  # for debugging
        assert len(w) == 2
        assert any("All formation energies for" in str(warn.message) for warn in w)
        assert any("You have not specified chemical potentials" in str(warn.message) for warn in w)
        return plot

    def test_format_defect_name(self):
        """
        Test ``format_defect_name()`` function.
        """
        # test standard behaviour
        formatted_name = plotting.format_defect_name(
            defect_species="vac_1_Cd_0",
            include_site_info_in_name=False,
        )
        assert formatted_name == "$\\it{V}\\!$ $_{Cd}^{0}$"
        # test with site number included
        formatted_name = plotting.format_defect_name(
            defect_species="vac_1_Cd_0",
            include_site_info_in_name=True,
        )
        assert formatted_name == "$\\it{V}\\!$ $_{Cd_{1}}^{0}$"

        # test interstitial case with site number excluded
        formatted_name = plotting.format_defect_name(
            defect_species="Int_Cd_1_0",
            include_site_info_in_name=False,
        )
        assert formatted_name == "Cd$_i^{0}$"
        # test interstitial case with site number included
        formatted_name = plotting.format_defect_name(
            defect_species="Int_Cd_1_0",
            include_site_info_in_name=True,
        )
        assert formatted_name == "Cd$_{i_{1}}^{0}$"

        # test lowercase interstitial with site number excluded
        formatted_name = plotting.format_defect_name(
            defect_species="int_Cd_1_0",
            include_site_info_in_name=False,
        )
        assert formatted_name == "Cd$_i^{0}$"
        # test lowercase interstitial with site number included
        formatted_name = plotting.format_defect_name(
            defect_species="int_Cd_1_0",
            include_site_info_in_name=True,
        )
        assert formatted_name == "Cd$_{i_{1}}^{0}$"

        # test uppercase vacancy (pymatgen default name) with site number excluded
        formatted_name = plotting.format_defect_name(
            defect_species="Vac_1_Cd_0",
            include_site_info_in_name=False,
        )
        assert formatted_name == "$\\it{V}\\!$ $_{Cd}^{0}$"
        # test uppercase vacancy (pymatgen default name) with site number included
        formatted_name = plotting.format_defect_name(
            defect_species="Vac_1_Cd_0",
            include_site_info_in_name=True,
        )
        assert formatted_name == "$\\it{V}\\!$ $_{Cd_{1}}^{0}$"

        # test substitution with site number excluded
        formatted_name = plotting.format_defect_name(
            defect_species="as_1_Ni_on_Li_0",
            include_site_info_in_name=False,
        )
        assert formatted_name == "Ni$_{Li}^{0}$"

        # test substitution with site number included
        formatted_name = plotting.format_defect_name(
            defect_species="as_1_Ni_on_Li_0",
            include_site_info_in_name=True,
        )
        assert formatted_name == "Ni$_{Li_{1}}^{0}$"

        # test substitution with site number excluded, current doped format, two-letter subbed element
        formatted_name = plotting.format_defect_name(
            defect_species="as_1_P_on_Na_-1",
            include_site_info_in_name=False,
        )
        assert formatted_name == "P$_{Na}^{-1}$"

        # test substitution with site number included, current doped format, two-letter subbed element
        formatted_name = plotting.format_defect_name(
            defect_species="as_1_P_on_Na_-1 ",
            include_site_info_in_name=True,
        )
        assert formatted_name == "P$_{Na_{1}}^{-1}$"

        # test substitution with site number excluded, current doped format
        formatted_name = plotting.format_defect_name(
            defect_species="as_2_Na_on_P_0",
            include_site_info_in_name=False,
        )
        assert formatted_name == "Na$_{P}^{0}$"

        # test substitution with site number included, current doped format
        formatted_name = plotting.format_defect_name(
            defect_species="as_2_Na_on_P_0",
            include_site_info_in_name=True,
        )
        assert formatted_name == "Na$_{P_{2}}^{0}$"

        # test interstitial with site number excluded, current doped format
        formatted_name = plotting.format_defect_name(
            defect_species="inter_12_P_0",
            include_site_info_in_name=False,
        )
        assert formatted_name == "P$_i^{0}$"

        # test interstitial with site number included, current doped format
        formatted_name = plotting.format_defect_name(
            defect_species="inter_12_P_0",
            include_site_info_in_name=True,
        )
        assert formatted_name == "P$_{i_{12}}^{0}$"

        # test vacancy with site number excluded, current doped format
        formatted_name = plotting.format_defect_name(
            defect_species="vac_4_P_-2",
            include_site_info_in_name=False,
        )
        assert formatted_name == "$\\it{V}\\!$ $_{P}^{-2}$"

        # test vacancy with site number included, current doped format
        formatted_name = plotting.format_defect_name(
            defect_species="vac_4_P_-2",
            include_site_info_in_name=True,
        )
        assert formatted_name == "$\\it{V}\\!$ $_{P_{4}}^{-2}$"

        # check exceptions raised: invalid charge or defect_species
        # test error catching:
        wrong_charge_error = ValueError(
            "Problem reading defect name vac_1_Cd_a, should end with charge state after underscore"
        )
        with pytest.raises(ValueError) as e:
            plotting.format_defect_name(defect_species="vac_1_Cd_a", include_site_info_in_name=True)
        assert str(wrong_charge_error) in str(e.value)

        pytest.raises(
            TypeError,
            plotting.format_defect_name,
            defect_species=2,
            include_site_info_in_name=True,
        )
        # check invalid defect type returns None
        assert (
            plotting.format_defect_name(defect_species="kk_Cd_1_0", include_site_info_in_name=True) is None
        )

        defect_species_name_dict = {
            "vac_Cd_mult32_0": "$\\it{V}\\!$ $_{Cd}^{0}$",
            "VSb_0": "$\\it{V}\\!$ $_{Sb}^{0}$",
            "VI_9": "$\\it{V}\\!$ $_{I}^{+9}$",
            "V_Sb_0": "$\\it{V}\\!$ $_{Sb}^{0}$",
            "V_I,_-2": "$\\it{V}\\!$ $_{I}^{-2}$",
            "V_I_-2": "$\\it{V}\\!$ $_{I}^{-2}$",
            "VacSb_2": "$\\it{V}\\!$ $_{Sb}^{+2}$",
            "VacI_2": "$\\it{V}\\!$ $_{I}^{+2}$",
            "Vac_Sb_3": "$\\it{V}\\!$ $_{Sb}^{+3}$",
            "Vac_I_1": "$\\it{V}\\!$ $_{I}^{+1}$",
            "VaSb_3": "$\\it{V}\\!$ $_{Sb}^{+3}$",
            "VaI_9": "$\\it{V}\\!$ $_{I}^{+9}$",
            "Va_Sb_10": "$\\it{V}\\!$ $_{Sb}^{+10}$",
            "Va_I_4": "$\\it{V}\\!$ $_{I}^{+4}$",
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
            "Vac_Li_mult32_-1": "$\\it{V}\\!$ $_{Li}^{-1}$",
            "Vac_Li_mult32_-2": "$\\it{V}\\!$ $_{Li}^{-2}$",
            "Vac_Li_mult32_0": "$\\it{V}\\!$ $_{Li}^{0}$",
            "Vac_Li_mult32_+1": "$\\it{V}\\!$ $_{Li}^{+1}$",
            "v_Cd_-1": "$\\it{V}\\!$ $_{Cd}^{-1}$",
            "v_Te_+2": "$\\it{V}\\!$ $_{Te}^{+2}$",
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
            "inter_8_O_-1": "O$_i^{-1}$",
            "inter_14_O_0": "O$_i^{0}$",
            "inter_14_O_+2": "O$_i^{+2}$",
            "inter_14_Th_+2": "Th$_i^{+2}$",
            "vac_14_Th_+2": "$\\it{V}\\!$ $_{Th}^{+2}$",
            "As_i_C2_-3": "As$_i^{-3}$",
        }

        for defect_species, expected_name in defect_species_name_dict.items():
            formatted_name = plotting.format_defect_name(
                defect_species=defect_species,
                include_site_info_in_name=False,
            )
            assert formatted_name == expected_name

        defect_species_w_site_info_name_dict = {
            "vac_Cd_mult32_0": "$\\it{V}\\!$ $_{Cd_{m32}}^{0}$",
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
            "Vac_Li_mult32_-1": "$\\it{V}\\!$ $_{Li_{m32}}^{-1}$",
            "Vac_Li_mult32_-2": "$\\it{V}\\!$ $_{Li_{m32}}^{-2}$",
            "Vac_Li_mult32_0": "$\\it{V}\\!$ $_{Li_{m32}}^{0}$",
            "Vac_Li_mult32_1": "$\\it{V}\\!$ $_{Li_{m32}}^{+1}$",
            "v_Cd_-1": "$\\it{V}\\!$ $_{Cd}^{-1}$",
            "v_Te_2": "$\\it{V}\\!$ $_{Te}^{+2}$",
            "v_Tea_2": "$\\it{V}\\!$ $_{Te}^{+2}$",
            "Te_Cd_s32_2": "Te$_{Cd_{s32}}^{+2}$",
            "Te_Cd_s32c_2": "Te$_{Cd_{s32c}}^{+2}$",
            "Cd_Te_+2": "Cd$_{Te}^{+2}$",
            "Cd_Te_2": "Cd$_{Te}^{+2}$",
            "Cd_i_C3v_+2": "Cd$_{i_{C_{3v}}}^{+2}$",
            "Cd_i_Td_Te2.83_2": "Cd$_{i_{T_{d}-Te2.83}}^{+2}$",
            "Te_i_Td_Cd2.83_-2": "Te$_{i_{T_{d}-Cd2.83}}^{-2}$",
            "v_O_C3v_0": "$\\it{V}\\!$ $_{O_{C_{3v}}}^{0}$",
            "v_O_C3v_1": "$\\it{V}\\!$ $_{O_{C_{3v}}}^{+1}$",
            "v_O_D4h_2": "$\\it{V}\\!$ $_{O_{D_{4h}}}^{+2}$",
            "Y_O_D4h_0": "Y$_{O_{D_{4h}}}^{0}$",
            "Ti_Y_+2": "Ti$_{Y}^{+2}$",
            "Ti_O_D4h_0": "Ti$_{O_{D_{4h}}}^{0}$",
            "Y_i_C4v_O1.92_0": "Y$_{i_{C_{4v}-O1.92}}^{0}$",
            "Y_i_C4v_O2.68_1": "Y$_{i_{C_{4v}-O2.68}}^{+1}$",
            "Y_i_Cs_O1.71_-2": "Y$_{i_{C_{s}-O1.71}}^{-2}$",
            "Y_i_Cs_O1.95_2": "Y$_{i_{C_{s}-O1.95}}^{+2}$",
            "Y_i_D2d_0": "Y$_{i_{D_{2d}}}^{0}$",
            "v_O_C1_0": "$\\it{V}\\!$ $_{O_{C_{1}}}^{0}$",
            "v_O_C3_8": "$\\it{V}\\!$ $_{O_{C_{3}}}^{+8}$",
            "Li_i_C1_Li1.75_0": "Li$_{i_{C_{1}-Li1.75}}^{0}$",
            "Li_i_C1_O1.72_1": "Li$_{i_{C_{1}-O1.72}}^{+1}$",
            "Li_i_C1_O1.78_2": "Li$_{i_{C_{1}-O1.78}}^{+2}$",
            "Li_i_C2_Li1.84O1.84_0": "Li$_{i_{C_{2}-Li1.84O1.84}}^{0}$",
            "Li_i_C2_Li1.84O1.94_1": "Li$_{i_{C_{2}-Li1.84O1.94}}^{+1}$",
            "Li_i_C2_Li1.86_2": "Li$_{i_{C_{2}-Li1.86}}^{+2}$",
            "Li_i_C3_0": "Li$_{i_{C_{3}}}^{0}$",
            "Cu_i_Oh_0": "Cu$_{i_{O_{h}}}^{0}$",
            "Cu_i_Td_0": "Cu$_{i_{T_{d}}}^{0}$",
            "Cu_i_C3v_Ag1.56Cu1.56Ag2.99a_0": "Cu$_{i_{C_{3v}-Ag1.56Cu1.56Ag2.99a}}^{0}$",
            "Cu_i_C3v_Ag1.56Cu1.56Ag2.99b_0": "Cu$_{i_{C_{3v}-Ag1.56Cu1.56Ag2.99b}}^{0}$",
            "Cu_i_C3v_Ag1.80_0": "Cu$_{i_{C_{3v}-Ag1.80}}^{0}$",
            "Te_i_C1_Cd2.71Te2.71Cd4.25j_-2": "Te$_{i_{C_{1}-Cd2.71Te2.71Cd4.25j}}^{-2}$",
            "Te_i_C1_Cd2.71Te2.71Cd4.25l_-2": "Te$_{i_{C_{1}-Cd2.71Te2.71Cd4.25l}}^{-2}$",
            "v_Cd_C1_Te2.83Cd4.62Te5.42a_-1": "$\\it{V}\\!$ $_{Cd_{C_{1}-Te2.83Cd4.62Te5.42a}}^{-1}$",
            "v_Cd_C1_Te2.83Cd4.62Te5.42b_-1": "$\\it{V}\\!$ $_{Cd_{C_{1}-Te2.83Cd4.62Te5.42b}}^{-1}$",
            "v_Cd_C3v_Cd2.71_-1": "$\\it{V}\\!$ $_{Cd_{C_{3v}-Cd2.71}}^{-1}$",
            "v_Cd_C3v_Te2.83Cd4.25_-1": "$\\it{V}\\!$ $_{Cd_{C_{3v}-Te2.83Cd4.25}}^{-1}$",
            "v_Cd_C3v_Te2.83Cd4.62_-1": "$\\it{V}\\!$ $_{Cd_{C_{3v}-Te2.83Cd4.62}}^{-1}$",
            "inter_8_O_-1": "O$_{i_{8}}^{-1}$",
            "inter_14_O_0": "O$_{i_{14}}^{0}$",
            "inter_14_O_+2": "O$_{i_{14}}^{+2}$",
            "inter_14_Th_+2": "Th$_{i_{14}}^{+2}$",
            "vac_14_Th_+2": "$\\it{V}\\!$ $_{Th_{14}}^{+2}$",
            "As_i_C2_-3": "As$_{i_{C_{2}}}^{-3}$",
        }
        for (
            defect_species,
            expected_name,
        ) in defect_species_w_site_info_name_dict.items():
            formatted_name = plotting.format_defect_name(
                defect_species=defect_species,
                include_site_info_in_name=True,
            )
            assert formatted_name == expected_name

    @custom_mpl_image_compare(filename="neutral_v_O_plot.png")
    def test_plot_neutral_v_O_V2O5(self):
        """
        Test plotting V2O5 defects, FNV correction.
        """
        dielectric = [4.186, 19.33, 17.49]
        bulk_path = f"{data_dir}/V2O5/V2O5_bulk"
        chempots = loadfn(f"{data_dir}/V2O5/chempots.json")

        defect_dict = {
            defect: analysis.defect_entry_from_paths(f"{data_dir}/V2O5/{defect}", bulk_path, dielectric)
            for defect in [dir for dir in os.listdir(f"{data_dir}/V2O5") if "v_O" in dir]
        }  # charge auto-determined (as neutral)
        thermo = DefectThermodynamics(list(defect_dict.values()))
        return thermo.plot(chempots, limit="V2O5-O2")


class DefectThermodynamicsPlotsTestCase(DefectThermodynamicsSetupMixin):
    def setUp(self):
        super().setUp()
        self.CdTe_LZ_thermo_wout_meta = deepcopy(self.orig_CdTe_LZ_thermo_wout_meta)
        self.Se_amalgamated_extrinsic_thermo = deepcopy(self.orig_Se_amalgamated_extrinsic_thermo)
        self.Bi_Se_thermo = deepcopy(self.orig_Bi_Se_thermo)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.CdTe_LZ_defect_dict = loadfn(
            os.path.join(data_dir, "CdTe_LZ_defect_dict_v2.3_wout_meta.json.gz")
        )
        cls.orig_CdTe_LZ_thermo_wout_meta = DefectThermodynamics(
            cls.CdTe_LZ_defect_dict, chempots=cls.CdTe_chempots
        )

        cls.orig_Se_amalgamated_extrinsic_thermo = DefectThermodynamics.from_json(
            f"{cls.EXAMPLE_DIR}/Se/Se_Amalgamated_Extrinsic_Thermo.json.gz"
        )
        cls.orig_Bi_Se_thermo = loadfn(f"{data_dir}/Bi_Se_BiSeI_thermo.json.gz")

    def test_plot_limit_no_chempots_error(self):
        with pytest.raises(ValueError) as exc:
            self.CdTe_defect_thermo.plot(limit="Te-rich")
        assert (
            "You have specified a chemical potential limit, but no `chempots` have been supplied, "
            "so `limit` cannot be used here!"
        ) in str(exc.value)

    def test_plot_limit_user_chempots_error(self):
        with pytest.raises(ValueError) as exc:
            self.CdTe_defect_thermo.plot(chempots={"Cd": 0.5, "Te": -0.5}, limit="Te-rich")
        assert (
            "You have specified a chemical potential limit, but the supplied chempots are not in the "
            "doped format (i.e. with `limits` in the chempots dict), and instead correspond to just a "
            "single chemical potential limit, so `limit` cannot be used here!"
        ) in str(exc.value)

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot_no_chempots.png")
    def test_default_CdTe_plot_no_chempots(self):
        with warnings.catch_warnings(record=True) as w:
            plot = self.CdTe_defect_thermo.plot()
        print([str(warn.message) for warn in w])  # for debugging
        assert len(w) == 2
        assert any("All formation energies for" in str(warn.message) for warn in w)
        assert any("You have not specified chemical potentials" in str(warn.message) for warn in w)
        return plot

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot(self):
        self.CdTe_defect_thermo.chempots = self.CdTe_chempots
        return self.CdTe_defect_thermo.plot(limit="Te-rich")

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot_explicit_limit(self):
        self.CdTe_defect_thermo.chempots = self.CdTe_chempots
        return self.CdTe_defect_thermo.plot(limit="CdTe-Te")

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot_specified_chempots(self):
        return self.CdTe_defect_thermo.plot(chempots=self.CdTe_chempots, limit="Te-rich")

    @custom_mpl_image_compare(filename="CdTe_manual_Te_rich_plot.png")
    def test_default_CdTe_plot_manual_chempots(self):
        return self.CdTe_defect_thermo.plot(
            chempots={"Cd": -1.25, "Te": 0}, el_refs=self.CdTe_chempots["elemental_refs"]
        )

    @custom_mpl_image_compare(filename="CdTe_manual_Te_rich_plot.png")
    def test_default_CdTe_plot_manual_chempots_at_init(self):
        defect_thermo = DefectThermodynamics(
            list(self.CdTe_defect_dict.values()),
            chempots={"Cd": -1.25, "Te": 0},
            el_refs=self.CdTe_chempots["elemental_refs"],
        )
        return defect_thermo.plot()

    @custom_mpl_image_compare(filename="CdTe_manual_Te_rich_plot.png")
    def test_default_CdTe_plot_manual_chempots_1by1(self):
        self.CdTe_defect_thermo.chempots = {"Cd": -1.25, "Te": 0}
        self.CdTe_defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        return self.CdTe_defect_thermo.plot()

    @custom_mpl_image_compare(filename="CdTe_manual_Te_rich_plot.png")
    def test_default_CdTe_plot_manual_chempots_1by1_other(self):
        self.CdTe_defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        self.CdTe_defect_thermo.chempots = {"Cd": -1.25, "Te": 0}
        return self.CdTe_defect_thermo.plot()

    @custom_mpl_image_compare(filename="CdTe_manual_Te_rich_plot.png")
    def test_default_CdTe_plot_manual_chempots_1by1_other2(self):
        self.CdTe_defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        return self.CdTe_defect_thermo.plot(chempots={"Cd": -1.25, "Te": 0})

    @custom_mpl_image_compare(filename="CdTe_manual_Te_rich_plot.png")
    def test_default_CdTe_plot_manual_chempots_1by1_other3(self):
        self.CdTe_defect_thermo.chempots = {"Cd": -1.25, "Te": 0}
        return self.CdTe_defect_thermo.plot(el_refs=self.CdTe_chempots["elemental_refs"])

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot_edited_el_refs(self):
        self.CdTe_defect_thermo.chempots = self.CdTe_chempots
        self.CdTe_defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        return self.CdTe_defect_thermo.plot(limit="Te-rich")

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot.png")
    def test_default_CdTe_plot_edited_el_refs_other(self):
        self.CdTe_defect_thermo.el_refs = self.CdTe_chempots["elemental_refs"]
        self.CdTe_defect_thermo.chempots = self.CdTe_chempots
        return self.CdTe_defect_thermo.plot(limit="Te-rich")

    @custom_mpl_image_compare(filename="CdTe_example_defects_plot_Cd_rich.png")
    def test_default_CdTe_plot_Cd_rich(self):
        return self.CdTe_defect_thermo.plot(chempots=self.CdTe_chempots, limit="Cd-rich")

    @custom_mpl_image_compare(filename="neutral_v_O_plot.png")
    def test_V2O5_O_rich_plot(self):
        return self.V2O5_defect_thermo.plot(limit="O-rich")

    @custom_mpl_image_compare(filename="neutral_v_O_plot.png")
    def test_V2O5_O_rich_plot_reduced_dist_tol(self):
        self.V2O5_defect_thermo.dist_tol = 1e-4
        # still two neutral vacancies merged as their site is the exact same
        return self.V2O5_defect_thermo.plot(limit="O-rich")

    @custom_mpl_image_compare(filename="neutral_v_O_plot_all_entries.png")
    def test_V2O5_O_rich_all_entries_plot(self):
        return self.V2O5_defect_thermo.plot(limit="O-rich", all_entries=True)

    @custom_mpl_image_compare(filename="neutral_v_O_plot_faded.png")
    def test_V2O5_O_rich_faded_plot(self):
        return self.V2O5_defect_thermo.plot(limit="O-rich", all_entries="faded")

    @custom_mpl_image_compare(filename="CdTe_LZ_all_default_Te_rich.png")
    def test_CdTe_LZ_all_defects_plot(self):
        return self.CdTe_LZ_thermo_wout_meta.plot(limit="Te-rich")

    @custom_mpl_image_compare(filename="CdTe_LZ_all_Te_rich_dist_tol_2.png")
    def test_CdTe_LZ_all_defects_plot_dist_tol_2(self):
        # Matches SK Thesis Fig 6.1b
        lz_cdte_defect_thermo = self.CdTe_LZ_thermo_wout_meta
        lz_cdte_defect_thermo.dist_tol = 2  # increase to 2 Å to merge Te_i defects

        return lz_cdte_defect_thermo.plot(chempots=self.CdTe_chempots, limit="Te-rich")

    @custom_mpl_image_compare(filename="CdTe_FNV_all_default_Te_rich_old_names.png")
    def test_CdTe_FNV_all_defects_plot_default_old_names(self):
        # Tests naming handling when old/unrecognised defect names are used
        cdte_defect_dict = loadfn(os.path.join(data_dir, "CdTe_defect_dict_old_names.json.gz"))
        cdte_defect_thermo = DefectThermodynamics(cdte_defect_dict)
        with warnings.catch_warnings(record=True) as w:
            plot = cdte_defect_thermo.plot()
        print([str(warn.message) for warn in w])  # for debugging
        assert any("You have not specified chemical potentials" in str(warn.message) for warn in w)
        assert any(
            "All formation energies for as_1_Te_on_Cd are below zero across the entire band gap range. "
            "This is typically unphysical (see docs), and likely due to mis-specification of chemical "
            "potentials (see docstrings and/or tutorials)." in str(warn.message)
            for warn in w
        )
        assert any("All formation energies for Int_Te_3 are below zero" in str(warn.message) for warn in w)
        assert len(w) == 3
        return plot

    @custom_mpl_image_compare(filename="Sb2O5_default_TLD.png")
    def test_Sb2O5_default(self):
        with warnings.catch_warnings(record=True) as w:
            plot = self.Sb2O5_defect_thermo.plot(limit="O-poor")
        print([str(warn.message) for warn in w])  # for debugging
        assert not w
        return plot

    @custom_mpl_image_compare(filename="Sb2O5_merged_dist_tol_TLD.png")
    def test_Sb2O5_merged_dist_tol(self):
        with warnings.catch_warnings(record=True) as w:
            self.Sb2O5_defect_thermo.dist_tol = 10
            plot = self.Sb2O5_defect_thermo.plot(limit="O-poor")
        print([str(warn.message) for warn in w])  # for debugging
        assert not w
        return plot

    @custom_mpl_image_compare(filename="CdTe_duplicate_entry_names.png")
    def test_handling_duplicate_entry_names_CdTe(self):
        """
        Test renaming behaviour when defect entries with the same names are
        provided.
        """
        defect_dict = loadfn(os.path.join(data_dir, "CdTe_defect_dict_v2.3.json.gz"))
        num_entries = len(defect_dict)
        for defect_entry in defect_dict.values():
            if "Cd_i" in defect_entry.name:
                defect_entry.name = f"Cd_i_{defect_entry.charge_state}"
            if "Te_i" in defect_entry.name:
                defect_entry.name = f"Te_i_{defect_entry.charge_state}"

        defect_thermo = DefectThermodynamics(defect_dict)
        assert len(defect_thermo.defect_entries) == num_entries
        print([entry.name for entry in defect_thermo.defect_entries])

        return defect_thermo.plot(self.CdTe_chempots, limit="Cd-rich")

    @custom_mpl_image_compare(filename="Se_duplicate_entry_names_old_plot.png")
    def test_handling_duplicate_entry_names_ext_Se_old_names(self):
        """
        Test renaming behaviour when defect entries with the same names are
        provided.

        In this case, the defect folder/entry names match the old ``doped``
        format, with e.g. ``sub_1_Br_on_Se_-1`` and ``inter_2_O_0`` etc.

        We have some duplicates (``inter_1_O_0`` and ``inter_2_O_0``) which
        are removed by including site info, but some (``inter_11_H_X``) which
        aren't, so ``_a`` & ``_b`` are appended to those names.
        """
        fig = self.Se_ext_no_pnict_thermo.plot()
        legend_txt = [t.get_text() for t in fig.get_axes()[0].get_legend().get_texts()]
        print(legend_txt)
        for i in ["O$_{i_{1}}$", "O$_{i_{2}}$", "H$_i$$_{-a}$", "H$_i$$_{-b}$"]:
            assert i in legend_txt

        return fig

    @custom_mpl_image_compare(filename="Se_pnictogen_plot.png")
    def test_plotting_ext_Se_new_names(self):
        """
        Test plotting behaviour with pnictogen impurities in Se.

        Previously this would append site info to the defect names by default,
        but this is not necessary here as there are no inequivalent
        substitution/interstitial sites, so check that the legend is as
        expected with no site info.
        """
        fig = self.Se_pnict_thermo.plot()
        legend_txt = [t.get_text() for t in fig.get_axes()[0].get_legend().get_texts()]
        print(legend_txt)
        for i in ["i_{", "{Se_{", "}}$", "-a", "-b"]:
            assert all(i not in text for text in legend_txt)

        return fig

    @custom_mpl_image_compare(filename="Se_site_info_old_names_plot.png")
    def test_ext_Se_old_names_include_site_info(self):
        """
        Test plotting extrinsic defects in Se with ``include_site_info=True``.

        In this case, the defect folder/entry names match the old ``doped``
        format, with e.g. ``sub_1_Br_on_Se_-1`` and ``inter_2_O_0`` etc.

        We have some duplicates (``inter_1_O_0`` and ``inter_2_O_0``) which
        are removed by including site info, but some (``inter_11_H_X``) which
        aren't, so ``_a`` & ``_b`` are appended to those names.
        """
        fig = self.Se_ext_no_pnict_thermo.plot(include_site_info=True)
        legend_txt = [t.get_text() for t in fig.get_axes()[0].get_legend().get_texts()]
        print(legend_txt)
        for i in [
            "O$_{i_{1}}$",
            "O$_{i_{2}}$",
            "Te$_{Se_{1}}$",
            "H$_{i_{11}}$$_{-a}$",
            "H$_{i_{11}}$$_{-b}$",
            "F$_{i_{1}}$",
        ]:
            assert i in legend_txt

        return fig

    @custom_mpl_image_compare(filename="Se_pnictogen_site_info_plot.png")
    def test_plotting_ext_Se_site_info(self):
        """
        Test plotting behaviour with pnictogen impurities in Se, with
        ``include_site_info=True``.
        """
        fig = self.Se_pnict_thermo.plot(include_site_info=True)
        legend_txt = [t.get_text() for t in fig.get_axes()[0].get_legend().get_texts()]
        print(legend_txt)
        # site info being added; C2 for interstitials, none for _Se (only one site)
        assert all(any(i in text for i in ["{C_{2}}}", "_{Se"]) for text in legend_txt)

        return fig

    @custom_mpl_image_compare(filename="Se_extrinsic_interstitials_linestyles_plot.png")
    def test_plotting_linestyles_and_colors_ext_Se(self):
        """
        Test plotting extrinsic interstitials in Se, using the ``linestyles``
        and ``colormap`` (with ``ListedColormap``) options.
        """
        from matplotlib import colormaps
        from matplotlib.colors import ListedColormap

        from doped.core import Interstitial

        amalgamated_Se_extrinsic_interstitials_thermo = DefectThermodynamics(
            defect_entries=[
                entry
                for entry in self.Se_amalgamated_extrinsic_thermo.defect_entries
                if isinstance(entry.defect, Interstitial)
            ],
            chempots=self.Se_amalgamated_extrinsic_thermo.chempots,
        )
        amalgamated_Se_extrinsic_interstitials_thermo.dist_tol = 2  # amalgamate Hi and Oi

        # H, N, P, As, Sb, O, S, Te, F, Cl, Br:
        colors = colormaps.get("Dark2").colors
        H_color = colors[4]
        pnict_color = colors[2]
        chalc_color = colors[1]
        halogen_color = colors[0]

        H_pnict_chalc_halogen_colormap = ListedColormap(
            [
                H_color,
                *[(*pnict_color, 1 - 0.2 * i) for i in range(4)],
                *[(*chalc_color, 1 - 0.3 * i) for i in range(3)],
                *[(*halogen_color, 1 - 0.3 * i) for i in range(3)],
            ]
        )
        linestyles = [  # solid for first of each group, then dashed, dotted, double dash
            "-",
            "-",
            "--",
            ":",
            "-.",
            "-",
            "--",
            ":",
            "-",
            "--",
            ":",
        ]

        return amalgamated_Se_extrinsic_interstitials_thermo.plot(
            chempot_table=False, colormap=H_pnict_chalc_halogen_colormap, linestyles=linestyles
        )

    @custom_mpl_image_compare(filename="CdTe_LZ_all_Te_rich_site_info.png")
    def test_CdTe_LZ_site_info_plot(self):
        """
        Test CdTe plotting behaviour with ``include_site_info=True``.
        """
        return self.CdTe_LZ_thermo_wout_meta.plot(limit="Te-rich", include_site_info=True)

    @custom_mpl_image_compare(filename="Bi_Se_BiSeI_Se_rich.png")
    def test_Bi_Se_all_entries_plot(self):
        """
        Test plotting all entries in Bi_Se thermo.

        Previously had issue where the colours did not match the appropriate
        lines.
        """
        return self.Bi_Se_thermo.plot(limit="Se-rich", all_entries=True)

    @custom_mpl_image_compare(filename="Bi_Se_BiSeI_Se_rich.png")
    def test_Bi_Se_all_entries_manual_colormap_plot(self):
        from doped.utils.plotting import ListedColormap, get_colormap

        cmap = get_colormap("tab10")
        cmap = ListedColormap([(*color, 0.75) for color in cmap.colors])
        return self.Bi_Se_thermo.plot(limit="Se-rich", all_entries=True, colormap=cmap)

    @custom_mpl_image_compare(filename="Bi_Se_BiSeI_Se_rich.png")
    def test_Bi_Se_all_entries_unrecognised_colormap_plot(self):
        with warnings.catch_warnings(record=True) as w:
            fig = self.Bi_Se_thermo.plot(limit="Se-rich", all_entries=True, colormap="Well shiii")
        print([str(warn.message) for warn in w])
        assert any("Colormap 'Well shiii' not found in" in str(warn.message) for warn in w)
        assert any("Defaulting to 'tab10' colormap" in str(warn.message) for warn in w)
        return fig
