# Note that this also implicitly tests most of the `parse_calculations.py` module

import os
import shutil
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from pymatgen.core.structure import Structure

from doped.analysis import defect_entry_from_paths
from doped.pycdt.utils.parse_calculations import (
    get_defect_type_and_composition_diff,
    get_defect_site_idxs_and_unrelaxed_structure,
)


def if_present_rm(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


class DopedParsingTestCase(unittest.TestCase):
    def setUp(self):
        self.module_path = os.path.dirname(os.path.abspath(__file__))
        self.EXAMPLE_DIR = os.path.join(self.module_path, "../examples")
        self.CDTE_EXAMPLE_DIR = os.path.join(self.module_path, "../examples/CdTe")
        self.YTOS_EXAMPLE_DIR = os.path.join(self.module_path, "../examples/YTOS")
        self.CDTE_BULK_DATA_DIR = os.path.join(
            self.CDTE_EXAMPLE_DIR, "Bulk_Supercell/vasp_ncl"
        )
        self.cdte_dielectric = np.array(
            [[9.13, 0, 0], [0.0, 9.13, 0], [0, 0, 9.13]]
        )  # CdTe

        self.ytos_dielectric = [  # from legacy Materials Project
            [40.71948719643814, -9.282128210266565e-14, 1.26076160303219e-14],
            [-9.301652644020242e-14, 40.71948719776858, 4.149879443489052e-14],
            [5.311743673463141e-15, 2.041077680836527e-14, 25.237620491130023],
        ]

        self.general_delocalization_warning = """
Note: Defects throwing a "delocalization analysis" warning may require a larger supercell for
accurate total energies. Recommended to look at the correction plots (i.e. run 
`get_correction_freysoldt(DefectEntry,...,plot=True)` from
`doped.corrections`) to visually determine if the charge 
correction scheme is still appropriate (replace 'freysoldt' with 'kumagai' if using anisotropic 
correction). You can also change the DefectCompatibility() tolerance settings via the 
`compatibility` parameter in `SingleDefectParser.from_paths()`."""

    def tearDown(self):
        if_present_rm("bulk_voronoi_nodes.json")

        if os.path.exists(
            f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/hidden_otcr.gz"
        ):
            shutil.move(
                f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/hidden_otcr.gz",
                f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/OUTCAR.gz",
            )

        if os.path.exists(f"{self.YTOS_EXAMPLE_DIR}/F_O_1/hidden_otcr.gz"):
            shutil.move(
                f"{self.YTOS_EXAMPLE_DIR}/F_O_1/hidden_otcr.gz",
                f"{self.YTOS_EXAMPLE_DIR}/F_O_1/OUTCAR.gz",
            )

        if_present_rm(f"{self.CDTE_EXAMPLE_DIR}/vac_1_Cd_-2/vasp_ncl/another_LOCPOT.gz")
        if_present_rm(f"{self.CDTE_BULK_DATA_DIR}/another_LOCPOT.gz")
        if_present_rm(f"{self.CDTE_BULK_DATA_DIR}/another_OUTCAR.gz")
        if_present_rm(
            f"{self.CDTE_EXAMPLE_DIR}/vac_1_Cd_-2/vasp_ncl/another_vasprun.xml.gz"
        )
        if_present_rm(f"{self.CDTE_BULK_DATA_DIR}/another_vasprun.xml.gz")

        if os.path.exists(
            f"{self.CDTE_EXAMPLE_DIR}/vac_1_Cd_-2/vasp_ncl/hidden_lcpt.gz"
        ):
            shutil.move(
                f"{self.CDTE_EXAMPLE_DIR}/vac_1_Cd_-2/vasp_ncl/hidden_lcpt.gz",
                f"{self.CDTE_EXAMPLE_DIR}/vac_1_Cd_-2/vasp_ncl/LOCPOT.gz",
            )

        if_present_rm(f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/LOCPOT.gz")

    # test_auto_charge_determination -> done locally (and indirectly with example notebook) due
    # to non-availability of `POTCAR`s on GH Actions

    def test_auto_charge_correction_behaviour(self):
        """Test skipping of charge corrections and warnings"""
        defect_path = f"{self.CDTE_EXAMPLE_DIR}/vac_1_Cd_-2/vasp_ncl"
        fake_aniso_dielectric = [1, 2, 3]

        with warnings.catch_warnings(record=True) as w:
            parsed_v_cd_m2_fake_aniso = defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=self.CDTE_BULK_DATA_DIR,
                dielectric=fake_aniso_dielectric,
                charge=-2,
            )
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertIn(
                "An anisotropic dielectric constant was supplied, but `OUTCAR` files (needed to "
                "compute the _anisotropic_ Kumagai eFNV charge correction) were not found in the "
                f"defect (at {defect_path}) & bulk (at "
                f"{self.CDTE_BULK_DATA_DIR}) folders.\n`LOCPOT` files were found in both defect & "
                f"bulk folders, and so the Freysoldt (FNV) charge correction developed for "
                f"_isotropic_ materials will be applied here, which corresponds to using the "
                f"effective isotropic average of the supplied anisotropic dielectric. This could "
                f"lead to significant errors for very anisotropic systems and/or relatively small "
                f"supercells!",
                str(w[-1].message),
            )

        self.assertAlmostEqual(
            parsed_v_cd_m2_fake_aniso.uncorrected_energy, 7.661, places=3
        )
        self.assertAlmostEqual(
            parsed_v_cd_m2_fake_aniso.energy, 10.379714081555262, places=3
        )

        # test no warnings when skip_corrections is True
        with warnings.catch_warnings(record=True) as w:
            parsed_v_cd_m2_fake_aniso = defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=self.CDTE_BULK_DATA_DIR,
                dielectric=fake_aniso_dielectric,
                skip_corrections=True,
                charge=-2,
            )
            self.assertEqual(len(w), 0)

        self.assertAlmostEqual(
            parsed_v_cd_m2_fake_aniso.uncorrected_energy, 7.661, places=3
        )
        self.assertAlmostEqual(parsed_v_cd_m2_fake_aniso.energy, 7.661, places=3)
        self.assertEqual(parsed_v_cd_m2_fake_aniso.corrections, {})

        # test isotropic dielectric but only OUTCAR present:
        with warnings.catch_warnings(record=True) as w:
            parsed_int_Te_2_fake_aniso = defect_entry_from_paths(
                defect_path=f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl",
                bulk_path=self.CDTE_BULK_DATA_DIR,
                dielectric=fake_aniso_dielectric,
                charge=2,
            )
            self.assertIn(
                f"Multiple `OUTCAR` files found in defect directory: "
                f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl. Using "
                f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/OUTCAR.gz to parse core levels and "
                f"compute the Kumagai (eFNV) image charge correction.",
                str(w[0].message),
            )
            self.assertIn(
                f"Delocalization analysis has indicated that {parsed_int_Te_2_fake_aniso.name} "
                f"with charge +2 may not be compatible with the chosen charge correction.",
                str(w[1].message),
            )
            if len(w) == 3:  # depends on run ordering on GH Actions
                self.assertIn(self.general_delocalization_warning, str(w[2].message))

        self.assertAlmostEqual(
            parsed_int_Te_2_fake_aniso.uncorrected_energy, -7.105, places=3
        )
        self.assertAlmostEqual(parsed_int_Te_2_fake_aniso.energy, -5.022, places=3)

        with warnings.catch_warnings(record=True) as w:
            parsed_int_Te_2 = defect_entry_from_paths(
                defect_path=f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl",
                bulk_path=self.CDTE_BULK_DATA_DIR,
                dielectric=self.cdte_dielectric,
                charge=2,
            )
        self.assertEqual(
            len(w), 1
        )  # no charge correction warning with iso dielectric, parsing from OUTCARs, but multiple
        # OUTCARs present -> warning
        self.assertAlmostEqual(parsed_int_Te_2.uncorrected_energy, -7.105, places=3)
        self.assertAlmostEqual(parsed_int_Te_2.energy, -6.221, places=3)

        # test warning when only OUTCAR present but no core level info (ICORELEVEL != 0)
        shutil.move(
            f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/OUTCAR.gz",
            f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/hidden_otcr.gz",
        )

        with warnings.catch_warnings(record=True) as w:
            parsed_int_Te_2_fake_aniso = defect_entry_from_paths(
                defect_path=f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl",
                bulk_path=self.CDTE_BULK_DATA_DIR,
                dielectric=fake_aniso_dielectric,
                charge=2,
            )
            self.assertEqual(len(w), 1)
            self.assertTrue(
                all(issubclass(warning.category, UserWarning) for warning in w)
            )
            self.assertIn(
                f"An anisotropic dielectric constant was supplied, but `OUTCAR` files (needed to "
                f"compute the _anisotropic_ Kumagai eFNV charge correction) in the defect (at "
                f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl) & bulk (at "
                f"{self.CDTE_BULK_DATA_DIR}) folders were unable to be "
                f"parsed, giving the following error message:\n"
                f"Unable to parse atomic core potentials from defect `OUTCAR` at "
                f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/OUTCAR_no_core_levels.gz. This can "
                f"happen if `ICORELEVEL` was not set to 0 (= default) in the `INCAR`, or if the "
                f"calculation was finished prematurely with a `STOPCAR`. The Kumagai charge "
                f"correction cannot be computed without this data!\n-> Charge corrections will "
                f"not be applied for this defect.",
                str(w[0].message),
            )

        self.assertAlmostEqual(
            parsed_int_Te_2_fake_aniso.uncorrected_energy, -7.105, places=3
        )
        self.assertAlmostEqual(parsed_int_Te_2_fake_aniso.energy, -7.105, places=3)

        # test warning when no core level info in OUTCAR (ICORELEVEL != 0), but LOCPOT
        # files present, but anisotropic dielectric:
        shutil.copyfile(
            f"{self.CDTE_EXAMPLE_DIR}/vac_1_Cd_-2/vasp_ncl/LOCPOT.gz",
            f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/LOCPOT.gz",
        )

        with warnings.catch_warnings(record=True) as w:
            parsed_int_Te_2_fake_aniso = defect_entry_from_paths(
                defect_path=f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl",
                bulk_path=self.CDTE_BULK_DATA_DIR,
                dielectric=fake_aniso_dielectric,
                charge=2,
            )
            self.assertEqual(
                len(w), 2
            )  # now also with a delocalization analysis warning (using incorrect LOCPOT)
            self.assertTrue(
                all(issubclass(warning.category, UserWarning) for warning in w)
            )
            self.assertIn(
                f"An anisotropic dielectric constant was supplied, but `OUTCAR` files (needed to "
                f"compute the _anisotropic_ Kumagai eFNV charge correction) in the defect (at "
                f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl) & bulk (at "
                f"{self.CDTE_BULK_DATA_DIR}) folders were unable to be parsed, giving the "
                f"following error message:\n"
                f"Unable to parse atomic core potentials from defect `OUTCAR` at "
                f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/OUTCAR_no_core_levels.gz. This can "
                f"happen if `ICORELEVEL` was not set to 0 (= default) in the `INCAR`, or if the "
                f"calculation was finished prematurely with a `STOPCAR`. The Kumagai charge "
                f"correction cannot be computed without this data!\n"
                f"`LOCPOT` files were found in both defect & bulk folders, and so the Freysoldt ("
                f"FNV) charge correction developed for _isotropic_ materials will be applied "
                f"here, which corresponds to using the effective isotropic average of the supplied "
                f"anisotropic dielectric. This could lead to significant errors for very "
                f"anisotropic systems and/or relatively small supercells!",
                str(w[0].message),
            )

        self.assertAlmostEqual(
            parsed_int_Te_2_fake_aniso.uncorrected_energy, -7.105, places=3
        )
        self.assertAlmostEqual(parsed_int_Te_2_fake_aniso.energy, -4.734, places=3)

        if_present_rm(f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/LOCPOT.gz")

        # rename files back to original:
        shutil.move(
            f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/hidden_otcr.gz",
            f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/OUTCAR.gz",
        )

        # test warning when no OUTCAR or LOCPOT file found:
        defect_path = f"{self.CDTE_EXAMPLE_DIR}/vac_1_Cd_-2/vasp_ncl"
        shutil.move(
            f"{defect_path}/LOCPOT.gz",
            f"{defect_path}/hidden_lcpt.gz",
        )
        with warnings.catch_warnings(record=True) as w:
            parsed_v_cd_m2 = defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=self.CDTE_BULK_DATA_DIR,
                dielectric=self.cdte_dielectric,
                charge=-2,
            )
            self.assertEqual(len(w), 1)
            self.assertTrue(
                all(issubclass(warning.category, UserWarning) for warning in w)
            )
            self.assertIn(
                f"`LOCPOT` or `OUTCAR` files are not present in both the defect (at "
                f"{defect_path}) and bulk (at {self.CDTE_BULK_DATA_DIR}) folders. These are "
                f"needed to perform the finite-size charge corrections. Charge corrections will "
                f"not be applied for this defect.",
                str(w[0].message),
            )

        self.assertAlmostEqual(parsed_v_cd_m2.uncorrected_energy, 7.661, places=3)
        self.assertAlmostEqual(parsed_v_cd_m2.energy, 7.661, places=3)
        self.assertEqual(parsed_v_cd_m2.corrections, {})

        # move LOCPOT back to original:
        shutil.move(f"{defect_path}/hidden_lcpt.gz", f"{defect_path}/LOCPOT.gz")

        # test no warning when no OUTCAR or LOCPOT file found, but charge is zero:
        defect_path = f"{self.CDTE_EXAMPLE_DIR}/vac_1_Cd_0/vasp_ncl"  # no LOCPOT/OUTCAR

        with warnings.catch_warnings(record=True) as w:
            parsed_v_cd_0 = defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=self.CDTE_BULK_DATA_DIR,
                dielectric=self.cdte_dielectric,
                charge=0,
            )
            self.assertEqual(len(w), 0)

        self.assertAlmostEqual(parsed_v_cd_0.uncorrected_energy, 4.166, places=3)
        self.assertAlmostEqual(parsed_v_cd_0.energy, 4.166, places=3)

    def test_multiple_outcars(self):
        shutil.copyfile(
            f"{self.CDTE_BULK_DATA_DIR}/OUTCAR.gz",
            f"{self.CDTE_BULK_DATA_DIR}/another_OUTCAR.gz",
        )
        fake_aniso_dielectric = [1, 2, 3]
        with warnings.catch_warnings(record=True) as w:
            parsed_int_Te_2_fake_aniso = defect_entry_from_paths(
                defect_path=f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl",
                bulk_path=self.CDTE_BULK_DATA_DIR,
                dielectric=fake_aniso_dielectric,
                charge=2,
            )
            self.assertEqual(
                3, len(w)
            )  # one delocalization warning (general one already give) and multiple OUTCARs (both
            # defect and bulk) – (this fails if run on it's own – len(w) -> 4)
            self.assertTrue(
                all(issubclass(warning.category, UserWarning) for warning in w)
            )
            self.assertIn(
                f"Multiple `OUTCAR` files found in bulk directory: "
                f"{self.CDTE_BULK_DATA_DIR}. Using {self.CDTE_BULK_DATA_DIR}/OUTCAR.gz to parse "
                f"core levels and compute the Kumagai (eFNV) image charge correction.",
                str(w[0].message),
            )
            self.assertIn(
                f"Multiple `OUTCAR` files found in defect directory: "
                f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl. Using "
                f"{self.CDTE_EXAMPLE_DIR}/Int_Te_3_2/vasp_ncl/OUTCAR.gz to parse core levels and "
                f"compute the Kumagai (eFNV) image charge correction.",
                str(w[1].message),
            )
            # other two warnings are delocalization warnings, already tested

    def test_multiple_locpots(self):
        defect_path = f"{self.CDTE_EXAMPLE_DIR}/vac_1_Cd_-2/vasp_ncl"

        shutil.copyfile(f"{defect_path}/LOCPOT.gz", f"{defect_path}/another_LOCPOT.gz")
        shutil.copyfile(
            f"{self.CDTE_BULK_DATA_DIR}/LOCPOT.gz",
            f"{self.CDTE_BULK_DATA_DIR}/another_LOCPOT.gz",
        )

        with warnings.catch_warnings(record=True) as w:
            parsed_v_cd_m2 = defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=self.CDTE_BULK_DATA_DIR,
                dielectric=self.cdte_dielectric,
                charge=-2,
            )
            self.assertEqual(len(w), 2)  # multiple LOCPOTs (both defect and bulk)
            self.assertTrue(
                all(issubclass(warning.category, UserWarning) for warning in w)
            )
            self.assertIn(
                f"Multiple `LOCPOT` files found in bulk directory: "
                f"{self.CDTE_BULK_DATA_DIR}. Using {self.CDTE_BULK_DATA_DIR}/LOCPOT.gz to parse "
                f"the electrostatic potential and compute the Freysoldt (FNV) charge correction.",
                str(w[0].message),
            )
            self.assertIn(
                f"Multiple `LOCPOT` files found in defect directory: {defect_path}. Using "
                f"{defect_path}/LOCPOT.gz to parse the electrostatic potential and compute the "
                f"Freysoldt (FNV) charge correction.",
                str(w[1].message),
            )

    def test_multiple_vaspruns(self):
        defect_path = f"{self.CDTE_EXAMPLE_DIR}/vac_1_Cd_-2/vasp_ncl"

        shutil.copyfile(
            f"{defect_path}/vasprun.xml.gz", f"{defect_path}/another_vasprun.xml.gz"
        )
        shutil.copyfile(
            f"{self.CDTE_BULK_DATA_DIR}/vasprun.xml.gz",
            f"{self.CDTE_BULK_DATA_DIR}/another_vasprun.xml.gz",
        )

        with warnings.catch_warnings(record=True) as w:
            parsed_v_cd_m2 = defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=self.CDTE_BULK_DATA_DIR,
                dielectric=self.cdte_dielectric,
                charge=-2,
            )
            self.assertEqual(
                len(w), 2
            )  # multiple `vasprun.xml`s (both defect and bulk)
            self.assertTrue(
                all(issubclass(warning.category, UserWarning) for warning in w)
            )
            self.assertIn(
                f"Multiple `vasprun.xml` files found in bulk directory: "
                f"{self.CDTE_BULK_DATA_DIR}. Using {self.CDTE_BULK_DATA_DIR}/vasprun.xml.gz to "
                f"parse the calculation energy and metadata.",
                str(w[0].message),
            )
            self.assertIn(
                f"Multiple `vasprun.xml` files found in defect directory: {defect_path}. Using "
                f"{defect_path}/vasprun.xml.gz to parse the calculation energy and metadata.",
                str(w[1].message),
            )

    def test_dielectric_initialisation(self):
        """
        Test that dielectric can be supplied as float or int or 3x1 array/list or 3x3 array/list
        """
        defect_path = f"{self.CDTE_EXAMPLE_DIR}/vac_1_Cd_-2/vasp_ncl"
        # get correct Freysoldt correction energy:
        parsed_v_cd_m2 = (
            defect_entry_from_paths(  # defect charge determined automatically
                defect_path=defect_path,
                bulk_path=self.CDTE_BULK_DATA_DIR,
                dielectric=self.cdte_dielectric,
                charge=-2,
            )
        )

        # Check that the correct Freysoldt correction is applied
        correct_correction_dict = {
            "charge_correction": 0.7376460317828045,
            "bandfilling_correction": -0.0,
            "bandedgeshifting_correction": 0.0,
        }
        for correction_name, correction_energy in correct_correction_dict.items():
            self.assertAlmostEqual(
                parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                places=3,
            )

        # test float
        new_parsed_v_cd_m2 = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=self.CDTE_BULK_DATA_DIR,
            dielectric=9.13,
            charge=-2,
        )
        for correction_name, correction_energy in correct_correction_dict.items():
            self.assertAlmostEqual(
                new_parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                places=3,
            )

        # test int
        new_parsed_v_cd_m2 = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=self.CDTE_BULK_DATA_DIR,
            dielectric=9,
            charge=-2,
        )
        for correction_name, correction_energy in correct_correction_dict.items():
            self.assertAlmostEqual(
                new_parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                places=1,  # change places to 1, because using int now so slightly off (0.006
                # difference)
            )

        # test 3x1 array
        new_parsed_v_cd_m2 = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=self.CDTE_BULK_DATA_DIR,
            dielectric=np.array([9.13, 9.13, 9.13]),
            charge=-2,
        )
        for correction_name, correction_energy in correct_correction_dict.items():
            self.assertAlmostEqual(
                new_parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                places=3,
            )

        # test 3x1 list
        new_parsed_v_cd_m2 = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=self.CDTE_BULK_DATA_DIR,
            dielectric=[9.13, 9.13, 9.13],
            charge=-2,
        )
        for correction_name, correction_energy in correct_correction_dict.items():
            self.assertAlmostEqual(
                new_parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                places=3,
            )

        # test 3x3 array
        new_parsed_v_cd_m2 = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=self.CDTE_BULK_DATA_DIR,
            dielectric=self.cdte_dielectric,
            charge=-2,
        )
        for correction_name, correction_energy in correct_correction_dict.items():
            self.assertAlmostEqual(
                new_parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                places=3,
            )

        # test 3x3 list
        new_parsed_v_cd_m2 = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=self.CDTE_BULK_DATA_DIR,
            dielectric=self.cdte_dielectric.tolist(),
            charge=-2,
        )
        for correction_name, correction_energy in correct_correction_dict.items():
            self.assertAlmostEqual(
                new_parsed_v_cd_m2.corrections[correction_name],
                correction_energy,
                places=3,
            )

    def test_vacancy_parsing_and_freysoldt(self):
        """Test parsing of Cd vacancy calculations and correct Freysoldt correction calculated"""
        parsed_vac_Cd_dict = {}

        for i in os.listdir(self.CDTE_EXAMPLE_DIR):
            if "vac_1_Cd" in i:  # loop folders and parse those with "vac_1_Cd" in name
                defect_path = f"{self.CDTE_EXAMPLE_DIR}/{i}/vasp_ncl"
                defect_charge = int(i[-2:].replace("_", ""))
                # parse with no transformation.json
                parsed_vac_Cd_dict[i] = defect_entry_from_paths(
                    defect_path=defect_path,
                    bulk_path=self.CDTE_BULK_DATA_DIR,
                    dielectric=self.cdte_dielectric,
                    charge=defect_charge,
                )  # Keep dictionary of parsed defect entries

        self.assertTrue(len(parsed_vac_Cd_dict) == 3)
        self.assertTrue(all(f"vac_1_Cd_{i}" in parsed_vac_Cd_dict for i in [0, -1, -2]))
        # Check that the correct Freysoldt correction is applied
        for name, energy, correction_dict in [
            (
                "vac_1_Cd_0",
                4.166,
                {},
            ),
            (
                "vac_1_Cd_-1",
                6.355,
                {
                    "charge_correction": 0.22517150393292082,
                    "bandfilling_correction": -0.0,
                    "bandedgeshifting_correction": 0.0,
                },
            ),
            (
                "vac_1_Cd_-2",
                8.398,
                {
                    "charge_correction": 0.7376460317828045,
                    "bandfilling_correction": -0.0,
                    "bandedgeshifting_correction": 0.0,
                },
            ),
        ]:
            self.assertAlmostEqual(parsed_vac_Cd_dict[name].energy, energy, places=3)
            for correction_name, correction_energy in correction_dict.items():
                self.assertAlmostEqual(
                    parsed_vac_Cd_dict[name].corrections[correction_name],
                    correction_energy,
                    places=3,
                )

            # assert auto-determined vacancy site is correct
            # should be: PeriodicSite: Cd (6.5434, 6.5434, 6.5434) [0.5000, 0.5000, 0.5000]
            if name == "vac_1_Cd_0":
                np.testing.assert_array_almost_equal(
                    parsed_vac_Cd_dict[name].site.frac_coords, [0.5, 0.5, 0.5]
                )
            else:
                np.testing.assert_array_almost_equal(
                    parsed_vac_Cd_dict[name].site.frac_coords, [0, 0, 0]
                )

    def test_interstitial_parsing_and_kumagai(self):
        """Test parsing of Te (split-)interstitial and Kumagai-Oba (eFNV) correction"""
        with patch("builtins.print") as mock_print:
            for i in os.listdir(self.CDTE_EXAMPLE_DIR):
                if "Int_Te" in i:  # loop folders and parse those with "Int_Te" in name
                    defect_path = f"{self.CDTE_EXAMPLE_DIR}/{i}/vasp_ncl"
                    defect_charge = int(i[-2:].replace("_", ""))
                    # parse with no transformation.json:
                    te_i_2_ent = defect_entry_from_paths(
                        defect_path=defect_path,
                        bulk_path=self.CDTE_BULK_DATA_DIR,
                        dielectric=self.cdte_dielectric,
                        charge=defect_charge,
                    )  # Keep dictionary of parsed defect entries

        mock_print.assert_called_once_with(
            "Saving parsed Voronoi sites (for interstitial site-matching) "
            "to bulk_voronoi_sites.json to speed up future parsing."
        )

        self.assertAlmostEqual(te_i_2_ent.energy, -6.221, places=3)
        self.assertAlmostEqual(te_i_2_ent.uncorrected_energy, -7.105, places=3)
        correction_dict = {
            "charge_correction": 0.8834518111049584,
            "bandfilling_correction": -0.0,
            "bandedgeshifting_correction": 0.0,
        }
        for correction_name, correction_energy in correction_dict.items():
            self.assertAlmostEqual(
                te_i_2_ent.corrections[correction_name], correction_energy, places=3
            )

        # assert auto-determined interstitial site is correct
        # should be: PeriodicSite: Te (12.2688, 12.2688, 8.9972) [0.9375, 0.9375, 0.6875]
        np.testing.assert_array_almost_equal(
            te_i_2_ent.site.frac_coords, [0.9375, 0.9375, 0.6875]
        )

        # run again to check parsing of previous Voronoi sites
        with patch("builtins.print") as mock_print:
            for i in os.listdir(self.CDTE_EXAMPLE_DIR):
                if "Int_Te" in i:  # loop folders and parse those with "Int_Te" in name
                    defect_path = f"{self.CDTE_EXAMPLE_DIR}/{i}/vasp_ncl"
                    defect_charge = int(i[-2:].replace("_", ""))
                    # parse with no transformation.json:
                    te_i_2_ent = defect_entry_from_paths(
                        defect_path=defect_path,
                        bulk_path=self.CDTE_BULK_DATA_DIR,
                        dielectric=self.cdte_dielectric,
                        charge=defect_charge,
                    )

        mock_print.assert_not_called()
        os.remove("bulk_voronoi_nodes.json")

    def test_substitution_parsing_and_kumagai(self):
        """Test parsing of Te_Cd_1 and Kumagai-Oba (eFNV) correction"""
        for i in os.listdir(self.CDTE_EXAMPLE_DIR):
            if "as_1_Te" in i:  # loop folders and parse those with "as_1_Te" in name
                defect_path = f"{self.CDTE_EXAMPLE_DIR}/{i}/vasp_ncl"
                defect_charge = int(i[-2:].replace("_", ""))
                # parse with no transformation.json:
                te_cd_1_ent = defect_entry_from_paths(
                    defect_path=defect_path,
                    bulk_path=self.CDTE_BULK_DATA_DIR,
                    dielectric=self.cdte_dielectric,
                    charge=defect_charge,
                )

        self.assertAlmostEqual(te_cd_1_ent.energy, -2.665996, places=3)
        self.assertAlmostEqual(te_cd_1_ent.uncorrected_energy, -2.906, places=3)
        correction_dict = {
            "charge_correction": 0.24005014473002428,
            "bandfilling_correction": -0.0,
            "bandedgeshifting_correction": 0.0,
        }
        for correction_name, correction_energy in correction_dict.items():
            self.assertAlmostEqual(
                te_cd_1_ent.corrections[correction_name], correction_energy, places=3
            )
        # assert auto-determined substitution site is correct
        # should be: PeriodicSite: Te (6.5434, 6.5434, 6.5434) [0.5000, 0.5000, 0.5000]
        np.testing.assert_array_almost_equal(
            te_cd_1_ent.site.frac_coords, [0.5000, 0.5000, 0.5000]
        )

    def test_extrinsic_interstitial_defect_ID(self):
        """Test parsing of extrinsic F in YTOS interstitial"""
        bulk_sc_structure = Structure.from_file(f"{self.YTOS_EXAMPLE_DIR}/Bulk/POSCAR")
        initial_defect_structure = Structure.from_file(
            f"{self.YTOS_EXAMPLE_DIR}/Int_F_-1/Relaxed_CONTCAR"
        )
        (def_type, comp_diff) = get_defect_type_and_composition_diff(
            bulk_sc_structure, initial_defect_structure
        )
        self.assertEqual(def_type, "interstitial")
        self.assertDictEqual(comp_diff, {"F": 1})
        (
            bulk_site_idx,
            defect_site_idx,
            unrelaxed_defect_structure,
        ) = get_defect_site_idxs_and_unrelaxed_structure(
            bulk_sc_structure, initial_defect_structure, def_type, comp_diff
        )
        self.assertEqual(bulk_site_idx, None)
        self.assertEqual(defect_site_idx, len(unrelaxed_defect_structure) - 1)

        # assert auto-determined interstitial site is correct
        self.assertAlmostEqual(
            unrelaxed_defect_structure[
                defect_site_idx
            ].distance_and_image_from_frac_coords(
                [-0.0005726049122470, -0.0001544430438804, 0.47800736578014720]
            )[
                0
            ],
            0.0,
            places=2,
        )  # approx match, not exact because relaxed bulk supercell

    def test_extrinsic_substitution_defect_ID(self):
        """Test parsing of extrinsic U_on_Cd in CdTe"""
        bulk_sc_structure = Structure.from_file(
            f"{self.CDTE_EXAMPLE_DIR}/CdTe_bulk_supercell_POSCAR"
        )
        initial_defect_structure = Structure.from_file(
            f"{self.CDTE_EXAMPLE_DIR}/U_on_Cd_POSCAR"
        )
        (
            def_type,
            comp_diff,
        ) = get_defect_type_and_composition_diff(
            bulk_sc_structure, initial_defect_structure
        )
        self.assertEqual(def_type, "substitution")
        self.assertDictEqual(comp_diff, {"Cd": -1, "U": 1})
        (
            bulk_site_idx,
            defect_site_idx,
            unrelaxed_defect_structure,
        ) = get_defect_site_idxs_and_unrelaxed_structure(
            bulk_sc_structure, initial_defect_structure, def_type, comp_diff
        )
        self.assertEqual(bulk_site_idx, 0)
        self.assertEqual(defect_site_idx, 63)  # last site in structure

        # assert auto-determined substitution site is correct
        np.testing.assert_array_almost_equal(
            unrelaxed_defect_structure[defect_site_idx].frac_coords,
            [0.00, 0.00, 0.00],
            decimal=2,  # exact match because perfect supercell
        )

    def test_extrinsic_interstitial_parsing_and_kumagai(self):
        """Test parsing of extrinsic F in YTOS interstitial and Kumagai-Oba (eFNV) correction"""
        defect_path = f"{self.YTOS_EXAMPLE_DIR}/Int_F_-1/"
        # parse with no transformation.json or explicitly-set-charge:
        int_F_minus1_ent = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk/",
            dielectric=self.ytos_dielectric,
            charge=-1,
        )

        self.assertAlmostEqual(int_F_minus1_ent.energy, 0.767, places=3)
        self.assertAlmostEqual(int_F_minus1_ent.uncorrected_energy, 0.7515, places=3)
        correction_dict = {
            "charge_correction": 0.0155169495708003,
            "bandfilling_correction": -0.0,
            "bandedgeshifting_correction": 0.0,
        }
        for correction_name, correction_energy in correction_dict.items():
            self.assertAlmostEqual(
                int_F_minus1_ent.corrections[correction_name],
                correction_energy,
                places=3,
            )

        # assert auto-determined interstitial site is correct
        self.assertAlmostEqual(
            int_F_minus1_ent.site.distance_and_image_from_frac_coords([0, 0, 0.4847])[
                0
            ],
            0.0,
            places=2,
        )  # approx match, not exact because relaxed bulk supercell

        os.remove("bulk_voronoi_nodes.json")

    def test_extrinsic_substitution_parsing_and_freysoldt_and_kumagai(self):
        """
        Test parsing of extrinsic F-on-O substitution in YTOS, w/Kumagai-Oba (eFNV) and Freysoldt
        (FNV) corrections
        """

        # first using Freysoldt (FNV) correction – gives error because anisotropic dielectric
        defect_path = f"{self.YTOS_EXAMPLE_DIR}/F_O_1/"
        # hide OUTCAR file:
        shutil.move(f"{defect_path}/OUTCAR.gz", f"{defect_path}/hidden_otcr.gz")
        # parse with no transformation.json or explicitly-set-charge:
        F_O_1_ent = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk/",
            dielectric=self.ytos_dielectric,
            charge=1,
        )
        # move OUTCAR file back to original:
        shutil.move(f"{defect_path}/hidden_otcr.gz", f"{defect_path}/OUTCAR.gz")

        self.assertAlmostEqual(F_O_1_ent.energy, 0.03146836204627482, places=3)
        self.assertAlmostEqual(
            F_O_1_ent.uncorrected_energy, -0.08523418000004312, places=3
        )
        correction_dict = {
            "charge_correction": 0.11670254204631794,
            "bandfilling_correction": -0.0,
            "bandedgeshifting_correction": 0.0,
        }
        for correction_name, correction_energy in correction_dict.items():
            self.assertAlmostEqual(
                F_O_1_ent.corrections[correction_name], correction_energy, places=3
            )
        # assert auto-determined interstitial site is correct
        self.assertAlmostEqual(
            F_O_1_ent.site.distance_and_image_from_frac_coords([0, 0, 0])[0],
            0.0,
            places=2,
        )

        # now using Kumagai-Oba (eFNV) correction
        defect_path = f"{self.YTOS_EXAMPLE_DIR}/F_O_1/"
        # parse with no transformation.json or explicitly-set-charge:
        F_O_1_ent = defect_entry_from_paths(
            defect_path=defect_path,
            bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk/",
            dielectric=self.ytos_dielectric,
            charge=1,
        )

        self.assertAlmostEqual(F_O_1_ent.energy, -0.0031, places=3)
        self.assertAlmostEqual(F_O_1_ent.uncorrected_energy, -0.0852, places=3)
        correction_dict = {
            "charge_correction": 0.08214,
            "bandfilling_correction": -0.0,
            "bandedgeshifting_correction": 0.0,
        }
        for correction_name, correction_energy in correction_dict.items():
            self.assertAlmostEqual(
                F_O_1_ent.corrections[correction_name], correction_energy, places=3
            )
        # assert auto-determined interstitial site is correct
        self.assertAlmostEqual(
            F_O_1_ent.site.distance_and_image_from_frac_coords([0, 0, 0])[0],
            0.0,
            places=2,
        )

    def test_voronoi_structure_mismatch_and_reparse(self):
        """
        Test that a mismatch in bulk_supercell structure from previously parsed
        Voronoi nodes json file with current defect bulk supercell is detected and
        re-parsed
        """
        with patch("builtins.print") as mock_print:
            for i in os.listdir(self.CDTE_EXAMPLE_DIR):
                if "Int_Te" in i:  # loop folders and parse those with "Int_Te" in name
                    defect_path = f"{self.CDTE_EXAMPLE_DIR}/{i}/vasp_ncl"
                    # parse with no transformation.json or explicitly-set-charge:
                    te_i_2_ent = defect_entry_from_paths(
                        defect_path=defect_path,
                        bulk_path=self.CDTE_BULK_DATA_DIR,
                        dielectric=self.cdte_dielectric,
                        charge=2,
                    )

        mock_print.assert_called_once_with(
            "Saving parsed Voronoi sites (for interstitial site-matching) "
            "to bulk_voronoi_sites.json to speed up future parsing."
        )

        with warnings.catch_warnings(record=True) as w:
            defect_path = f"{self.YTOS_EXAMPLE_DIR}/Int_F_-1/"
            # parse with no transformation.json or explicitly-set-charge:
            int_F_minus1_ent = defect_entry_from_paths(
                defect_path=defect_path,
                bulk_path=f"{self.YTOS_EXAMPLE_DIR}/Bulk/",
                dielectric=self.ytos_dielectric,
                charge=-1,
            )

        warning_message = (
            "Previous bulk_voronoi_nodes.json detected, but does not "
            "match current bulk supercell. Recalculating Voronoi nodes."
        )
        user_warnings = [warning for warning in w if warning.category == UserWarning]
        self.assertEqual(len(user_warnings), 1)
        self.assertIn(warning_message, str(user_warnings[0].message))
        os.remove("bulk_voronoi_nodes.json")


if __name__ == "__main__":
    unittest.main()
