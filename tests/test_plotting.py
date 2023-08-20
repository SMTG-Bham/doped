"""
Tests for the `doped.analysis` module, which also implicitly tests most of the
`doped.pycdt.utils.parse_calculations` module.
"""

import os
import shutil
import unittest
from typing import Any, Dict

import numpy as np
import pytest

from doped import analysis
from doped.core import DefectEntry
from doped.utils.corrections import get_correction_freysoldt, get_correction_kumagai
from doped.utils.legacy_pmg.thermodynamics import DefectPhaseDiagram


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


class PlottingTestCase(unittest.TestCase):
    module_path: str
    example_dir: str
    cdte_example_dir: str
    ytos_example_dir: str
    cdte_bulk_data_dir: str
    cdte_dielectric: np.ndarray
    vac_Cd_dict: Dict[Any, Any]
    vac_Cd_dpd: DefectPhaseDiagram
    F_O_1_entry: DefectEntry

    @classmethod
    def setUpClass(cls) -> None:
        # prepare parsed defect data (from doped parsing example)
        cls.module_path = os.path.dirname(os.path.abspath(__file__))
        cls.example_dir = os.path.join(cls.module_path, "../examples")
        cls.cdte_example_dir = os.path.join(cls.module_path, "../examples/CdTe")
        cls.ytos_example_dir = os.path.join(cls.module_path, "../examples/YTOS")
        cls.cdte_bulk_data_dir = os.path.join(cls.cdte_example_dir, "Bulk_Supercell/vasp_ncl")
        cls.cdte_dielectric = np.array([[9.13, 0, 0], [0.0, 9.13, 0], [0, 0, 9.13]])  # CdTe

        cls.vac_Cd_dict = {}  # dictionary of parsed vacancy defect entries

        for i in os.listdir(cls.cdte_example_dir):  # loops through the example directory
            if "vac_1_Cd" in i:  # and parses folders that have "vac_1_Cd" in their name
                print(f"Parsing {i}...")
                defect_path = f"{cls.cdte_example_dir}/{i}/vasp_ncl"
                cls.vac_Cd_dict[i] = analysis.defect_entry_from_paths(
                    defect_path, cls.cdte_bulk_data_dir, cls.cdte_dielectric
                )

        cls.vac_Cd_dpd = analysis.dpd_from_defect_dict(cls.vac_Cd_dict)

        cls.F_O_1_entry = analysis.defect_entry_from_paths(
            defect_path=f"{cls.ytos_example_dir}/F_O_1",
            bulk_path=f"{cls.ytos_example_dir}/Bulk",
            dielectric=[40.7, 40.7, 25.2],
        )

    # def setUp(self):
    # self.parsed_vac_cd_dict = loadfn("parsed_vac_Cd_dict.json")

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
        return get_correction_freysoldt(self.vac_Cd_dict["vac_1_Cd_-2"], 9.13, plot=True, return_fig=True)

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
