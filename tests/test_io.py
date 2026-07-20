"""
Tests for the ``doped.io`` calculator input/output framework.

This includes ``CalculationOutputs``, calculator dispatch, and the deprecation
shims for pre-``doped.io`` import paths.
"""

import os
import unittest
import warnings

import numpy as np
import pytest
from test_utils import vasp_data_dir

from doped.io import get_calculation_outputs


class CalculationOutputsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.CdTe_corrections_dir = os.path.join(vasp_data_dir, "CdTe_charge_correction_tests")
        cls.defect_dir = f"{cls.CdTe_corrections_dir}/v_Cd_-2_vasp_gam"
        cls.bulk_dir = f"{cls.CdTe_corrections_dir}/bulk_vasp_gam"

    def test_get_calculation_outputs(self):
        """
        Test parsing a defect supercell calculation to ``CalculationOutputs``,
        with the ``doped.io`` calculator-dispatch function.
        """
        outputs = get_calculation_outputs(  # calculator="vasp" default
            self.defect_dir,
            load_planar_averaged_potentials=True,
            load_site_potentials=True,
            parse_projected_eigen=False,
        )
        assert outputs.calculator == "vasp"
        assert np.isclose(outputs.energy, -183.98, atol=0.02)
        assert len(outputs.structure) == 63
        assert outputs.nelect is not None
        assert set(outputs.planar_averaged_potentials.keys()) == {0, 1, 2}
        assert len(outputs.site_potentials) == len(outputs.structure)
        assert outputs.eigenvalues is not None
        assert {"incar", "kpoints", "potcar_symbols"}.issubset(outputs.run_metadata)

    def test_require(self):
        """
        Test the informative error for missing optional outputs.
        """
        outputs = get_calculation_outputs(self.bulk_dir, parse_projected_eigen=False)
        assert outputs.site_potentials is None
        with pytest.raises(ValueError, match=r"requires the .*site_potentials.* calculation output"):
            outputs.require("site_potentials", task="The eFNV (Kumagai) charge correction")
        outputs.require("energy", "structure")  # no error for populated attributes

    def test_corrections_from_calculation_outputs(self):
        """
        Test that the FNV & eFNV corrections computed from
        ``CalculationOutputs`` objects match those from direct output-file
        parsing.
        """
        from doped.analysis import DefectParser
        from doped.corrections import get_freysoldt_correction, get_kumagai_correction

        parser_kwargs = {"load_planar_averaged_potentials": True, "load_site_potentials": True}
        defect_outputs = get_calculation_outputs(
            self.defect_dir, parse_projected_eigen=False, **parser_kwargs
        )
        bulk_outputs = get_calculation_outputs(self.bulk_dir, parse_projected_eigen=False, **parser_kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # multiple-files warnings etc. not under test here
            defect_entry = DefectParser.from_paths(
                defect_path=self.defect_dir,
                bulk_path=self.bulk_dir,
                dielectric=9.13,
                parse_projected_eigen=False,
            ).defect_entry

            fnv_from_files = get_freysoldt_correction(
                defect_entry,
                defect_locpot=f"{self.defect_dir}/LOCPOT.gz",
                bulk_locpot=f"{self.bulk_dir}/LOCPOT.gz",
                verbose=False,
            )
            fnv_from_outputs = get_freysoldt_correction(
                defect_entry, defect_locpot=defect_outputs, bulk_locpot=bulk_outputs, verbose=False
            )
            assert np.isclose(fnv_from_files.correction_energy, fnv_from_outputs.correction_energy)
            # TODO: Also hard-test the value here

            efnv_from_files = get_kumagai_correction(
                defect_entry,
                defect_outcar=f"{self.defect_dir}/OUTCAR.gz",
                bulk_outcar=f"{self.bulk_dir}/OUTCAR.gz",
                verbose=False,
            )
            efnv_from_outputs = get_kumagai_correction(
                defect_entry, defect_outcar=defect_outputs, bulk_outcar=bulk_outputs, verbose=False
            )
            assert np.isclose(efnv_from_files.correction_energy, efnv_from_outputs.correction_energy)
            # TODO: Also hard-test the value here


class DeprecationShimsTestCase(unittest.TestCase):
    def test_doped_vasp_shim(self):
        """
        ``doped.vasp`` forwards to ``doped.io.vasp.inputs`` with a
        ``DeprecationWarning``.
        """
        import doped.io.vasp.inputs
        import doped.vasp

        with pytest.warns(DeprecationWarning, match="doped.vasp has moved to doped.io.vasp.inputs"):
            assert doped.vasp.DefectsSet is doped.io.vasp.inputs.DefectsSet

        with pytest.raises(AttributeError):
            _ = doped.vasp.definitely_not_a_real_attribute

    def test_parsing_shim(self):
        r"""
        The dissolved ``doped.utils.parsing`` module forwards its old names to
        their new homes with ``DeprecationWarning``\s.
        """
        import doped.io.vasp.outputs
        import doped.utils.mappings
        import doped.utils.parsing

        for name, module in [
            ("get_vasprun", doped.io.vasp.outputs),
            ("get_defect_type_and_composition_diff", doped.utils.mappings),
        ]:
            with pytest.warns(DeprecationWarning, match=f"{name} has moved to {module.__name__}"):
                assert getattr(doped.utils.parsing, name) is getattr(module, name)

        for private_name in ("_get_bulk_supercell", "_simple_spin_degeneracy_from_num_electrons"):
            with pytest.raises(AttributeError):  # private helpers are not aliased
                _ = getattr(doped.utils.parsing, private_name)

        with pytest.raises(AttributeError):
            _ = doped.utils.parsing.definitely_not_a_real_attribute
