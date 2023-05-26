"""
Tests for parsing
"""

import pytest
import os
import numpy as np
from pathlib import Path
from doped.pycdt.utils import parse_calculations

fileloc = Path(__file__).parent.parent / "tests/data/parsing/CdTe"


@pytest.fixture
def parse_dataset():
    """
    Parse defect calculation that are matched with the tags passed.
    """
    def parse(tags=None, freysoldt=False, kumagai=False, bulk_name="Bulk_Supercell/vasp_gam"):
        bulk_file_path = fileloc / bulk_name
        dielectric = np.array([[9.13, 0, 0],[0., 9.13, 0],[0, 0, 9.13]])
        parsed_dict = {}

        for i in os.listdir(fileloc): 
            if not tags or any(map(lambda x: x in i, tags)):

                defect_file_path = fileloc / f"{i}/vasp_gam"
                defect_charge = int(i[-2:].replace("_",""))

                sdp = parse_calculations.SingleDefectParser.from_paths(defect_file_path, bulk_file_path,
                                                    dielectric, defect_charge)
                if freysoldt:
                    sdp.freysoldt_loader()
                if kumagai:
                    sdp.kumagai_loader()

                sdp.get_stdrd_metadata()
                sdp.get_bulk_gap_data()
                sdp.run_compatibility()
                parsed_dict[i] = sdp.defect_entry 
        return parsed_dict
    return parse




def test_parsing_cdte(parse_dataset):
    """Test parsing CdTe example"""

    defect_dict = parse_dataset(['vac_1_Cd_-2'])
    assert 'vac_1_Cd_-2' in defect_dict
    expected_energy =  7.4475896
    assert defect_dict['vac_1_Cd_-2'].energy == pytest.approx(expected_energy, abs=1e-3)
    return


def test_kumagai_order(parse_dataset):
    """Test kumagai defect correction parser can handle mismatched atomic orders"""

    defect_dict_orig = parse_dataset(['vac_1_Cd_-2'], kumagai=True)
    defect_dict_alt = parse_dataset(['vac_1_Cd_-2'], bulk_name="Bulk_Supercell_alt/vasp_gam", kumagai=True)
    assert defect_dict_orig['vac_1_Cd_-2'].energy == pytest.approx(defect_dict_alt['vac_1_Cd_-2'].energy, abs=1e-6)

def test_freysoldt_order(parse_dataset):
    """Test kumagai defect correction parser can handle mismatched atomic orders"""

    defect_dict_orig = parse_dataset(['vac_1_Cd_-2'], freysoldt=True)
    defect_dict_alt = parse_dataset(['vac_1_Cd_-2'], bulk_name="Bulk_Supercell_alt/vasp_gam", freysoldt=True)
    assert defect_dict_orig['vac_1_Cd_-2'].energy == pytest.approx(defect_dict_alt['vac_1_Cd_-2'].energy, abs=1e-6)
