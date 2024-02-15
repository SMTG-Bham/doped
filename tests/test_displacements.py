"""
Tests for the `doped.utils.displacements` module.
"""

import os
import shutil
import unittest

import matplotlib as mpl
import numpy as np
from monty.serialization import loadfn
from test_thermodynamics import custom_mpl_image_compare, data_dir

from doped import core
from doped.utils.displacements import _calc_site_displacements

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


class DefectDisplacementsTestCase(unittest.TestCase):
    def setUp(self):
        self.module_path = os.path.dirname(os.path.abspath(__file__))
        self.EXAMPLE_DIR = os.path.join(self.module_path, "../examples")
        self.CdTe_EXAMPLE_DIR = os.path.join(self.module_path, "../examples/CdTe")
        self.CdTe_thermo = loadfn(f"{self.CdTe_EXAMPLE_DIR}/CdTe_example_thermo.json")
        self.CdTe_chempots = loadfn(f"{self.CdTe_EXAMPLE_DIR}/CdTe_chempots.json")
        self.YTOS_EXAMPLE_DIR = os.path.join(self.module_path, "../examples/YTOS")
        self.YTOS_thermo = loadfn(f"{self.YTOS_EXAMPLE_DIR}/YTOS_example_thermo.json")

    def test_calc_site_displacements(self):
        """
        Test _calc_site_displacements() function.
        """
        # Vacancy:
        defect_entry = core.DefectEntry.from_json(f"{data_dir}/v_Cd_defect_entry.json")
        disp_dict = _calc_site_displacements(defect_entry)
        for i, disp in [
            (0, [0.0572041, 0.00036486, -0.01794981]),
            (15, [0.11715445, -0.03659073, 0.01312027]),
        ]:
            np.allclose(disp_dict["Abs. displacement"][i], np.array(disp))
        # Substitution:
        defect_entry = core.DefectEntry.from_json(f"{data_dir}/Te_Cd_+1_defect_entry.json")
        disp_dict = _calc_site_displacements(defect_entry)
        for i, disp in [
            (0, [0.00820645, 0.00821417, -0.00815738]),
            (15, [-0.00639524, 0.00639969, -0.01407927]),
        ]:
            np.allclose(disp_dict["Abs. displacement"][i], np.array(disp))
        # Interstitial:
        defect_entry = core.DefectEntry.from_json(f"{data_dir}/Int_Te_3_1_defect_entry.json")
        disp_dict = _calc_site_displacements(defect_entry)
        for i, disp in [
            (0, [-0.03931121, 0.01800569, 0.04547194]),
            (15, [-0.04850126, -0.01378455, 0.05439607]),
        ]:
            np.allclose(disp_dict["Abs. displacement"][i], np.array(disp))

    @custom_mpl_image_compare(filename="v_Cd_0_disp_proj_plot.png")
    def test_plot_site_displacements_proj(self):
        # Vacancy, displacement separated by direction:
        defect_entry = core.DefectEntry.from_json(f"{data_dir}/v_Cd_defect_entry.json")
        return defect_entry.plot_site_displacements(separated_by_direction=True, use_plotly=False)

    @custom_mpl_image_compare(filename="v_Cd_0_disp_plot.png")
    def test_plot_site_displacements(self):
        # Vacancy, total displacement
        defect_entry = core.DefectEntry.from_json(f"{data_dir}/v_Cd_defect_entry.json")
        return defect_entry.plot_site_displacements(separated_by_direction=False, use_plotly=False)

    @custom_mpl_image_compare(filename="YTOS_Int_F_-1_site_displacements.png")
    def test_plot_site_displacements_ytos(self):
        # Vacancy, total displacement
        defect_entry = core.DefectEntry.from_json(f"{data_dir}/YTOS_Int_F_-1_defect_entry.json")
        return defect_entry.plot_site_displacements(separated_by_direction=True, use_plotly=False)
