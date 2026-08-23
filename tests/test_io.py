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
from pymatgen.electronic_structure.core import Spin
from test_utils import vasp_data_dir

from doped.io import get_calculation_outputs
from doped.io.outputs import nelect_from_eigenvalues


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
        assert outputs.charge == -2  # auto-determined from NELECT vs neutral electron count
        assert outputs.spin_degeneracy() == 1  # even-electron (closed-shell) system
        assert outputs.spin_degeneracy(charge_state=-2) == 1

    def test_raw_objects_and_serialisation(self):
        """
        Test the (non-serialised) ``raw`` calculator objects dict,
        ``get_computed_entry()``, and ``as_dict()``/``from_dict()`` round-
        tripping.
        """
        from pymatgen.io.vasp.outputs import Vasprun

        from doped.io.outputs import CalculationOutputs

        outputs = get_calculation_outputs(self.defect_dir, parse_projected_eigen=False)
        assert isinstance(outputs.raw["vasprun"], Vasprun)
        computed_entry = outputs.get_computed_entry()
        assert computed_entry is outputs.raw["computed_entry"]
        assert np.isclose(computed_entry.energy, outputs.energy)
        assert computed_entry.parameters  # VASP-parsed entry with calculation parameters

        dct = outputs.as_dict()
        assert "raw" not in dct
        assert outputs.raw  # not clobbered by serialisation
        reloaded = CalculationOutputs.from_dict(dct)
        assert np.isclose(reloaded.energy, outputs.energy)
        assert len(reloaded.structure) == len(outputs.structure)
        assert reloaded.raw == {}  # raw objects not serialised
        assert np.isclose(reloaded.get_computed_entry().energy, outputs.energy)  # bare-entry fallback

    def test_require(self):
        """
        Test the informative error for missing optional outputs.
        """
        outputs = get_calculation_outputs(self.bulk_dir, parse_projected_eigen=False)
        assert outputs.site_potentials is None
        with pytest.raises(ValueError, match=r"requires the .*site_potentials.* calculation output"):
            outputs.require("site_potentials", task="The eFNV (Kumagai) charge correction")
        outputs.require("energy", "structure")  # no error for populated attributes

    def test_nelect_from_eigenvalues(self):
        """
        Test the ``nelect_from_eigenvalues`` fallback for determining the
        electron count from the band occupancies (used when the electron count
        is not directly available, e.g. ``NELECT`` with VASP), for the
        different occupancy conventions.
        """
        outputs = get_calculation_outputs(self.defect_dir, parse_projected_eigen=False)
        assert nelect_from_eigenvalues(outputs.eigenvalues, outputs.kpoint_weights) == outputs.nelect

        # one spin channel with singly-normalised occupancies (as with VASP) is doubled, unless
        # non-collinear (i.e. one electron per spinor band); 1 k-point, 3 bands:
        eig_occs = {Spin.up: np.array([[[-1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]])}
        assert nelect_from_eigenvalues(eig_occs, [1.0]) == 4.0
        assert nelect_from_eigenvalues(eig_occs, [1.0], noncollinear=True) == 2.0

        # but doubly-occupied bands (as written by some calculators) are not doubled:
        assert nelect_from_eigenvalues({Spin.up: np.array([[[-1.0, 2.0], [0.0, 2.0]]])}, [1.0]) == 4.0

        # and neither are (collinear) spin-polarised calculations, which are summed over both channels:
        assert (
            nelect_from_eigenvalues(dict.fromkeys((Spin.up, Spin.down), eig_occs[Spin.up]), [1.0]) == 4.0
        )

    def test_corrections_from_calculation_outputs(self):
        """
        Test that the FNV & eFNV corrections computed from
        ``CalculationOutputs`` objects match those from direct output-file
        parsing.
        """
        from doped.corrections import get_freysoldt_correction, get_kumagai_correction
        from doped.parsing import DefectParser

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


class SerializedBackendTestCase(unittest.TestCase):
    """
    Test the ``doped.io.serialized`` escape-hatch backend, which parses pre-
    serialised ``CalculationOutputs`` JSON files (i.e. the calculator-agnostic
    parsing pathway, usable with any calculator).
    """

    @classmethod
    def setUpClass(cls):
        cls.CdTe_corrections_dir = os.path.join(vasp_data_dir, "CdTe_charge_correction_tests")
        cls.defect_dir = f"{cls.CdTe_corrections_dir}/v_Cd_-2_vasp_gam"
        cls.bulk_dir = f"{cls.CdTe_corrections_dir}/bulk_vasp_gam"
        cls.parse_kwargs = {
            "load_planar_averaged_potentials": True,
            "load_site_potentials": True,
            "parse_projected_eigen": False,
        }

    def _serialise_outputs_to_tmp_tree(self, tmpdir):
        """
        Build ``CalculationOutputs`` (with the VASP backend), and serialise to
        a ``{tmpdir}/{v_Cd_-2,CdTe_bulk}/calculation_outputs.json.gz`` tree.
        """
        from monty.serialization import dumpfn

        folders = {}
        for name, src_dir in [("v_Cd_-2", self.defect_dir), ("CdTe_bulk", self.bulk_dir)]:
            outputs = get_calculation_outputs(src_dir, **self.parse_kwargs)
            folder = os.path.join(tmpdir, name)
            os.makedirs(folder)
            dumpfn(outputs, os.path.join(folder, "calculation_outputs.json.gz"))
            folders[name] = folder
        return folders

    def test_serialized_backend_json_roundtrip(self):
        """
        ``CalculationOutputs`` survive a full JSON file round-trip (including
        ``Spin``-keyed eigenvalues and axis-keyed potentials).
        """
        import tempfile

        from doped.io.outputs import CalculationOutputs

        outputs = get_calculation_outputs(self.defect_dir, **self.parse_kwargs)
        with tempfile.TemporaryDirectory() as tmpdir:
            folders = self._serialise_outputs_to_tmp_tree(tmpdir)
            reloaded = get_calculation_outputs(folders["v_Cd_-2"], calculator="serialized")

        assert isinstance(reloaded, CalculationOutputs)
        assert np.isclose(reloaded.energy, outputs.energy)
        assert reloaded.charge == outputs.charge == -2
        assert len(reloaded.structure) == len(outputs.structure)
        assert set(reloaded.eigenvalues.keys()) == set(outputs.eigenvalues.keys())  # Spin keys restored
        for spin, array in outputs.eigenvalues.items():
            assert np.allclose(reloaded.eigenvalues[spin], array)
        assert set(reloaded.planar_averaged_potentials.keys()) == {0, 1, 2}
        assert np.allclose(reloaded.site_potentials, outputs.site_potentials)
        assert reloaded.raw == {}

    def test_serialized_backend_defect_parsing(self):
        """
        ``DefectsParser``/``DefectParser`` parsing with the ``serialized``
        backend matches direct VASP output-file parsing (energies, charge
        states, charge corrections & band edge data).
        """
        import tempfile

        from doped.parsing import DefectParser, DefectsParser

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # multiple-files & SnB deprecation warnings not under test
            reference_entry = DefectParser.from_paths(  # direct VASP output-file parsing
                defect_path=self.defect_dir,
                bulk_path=self.bulk_dir,
                dielectric=9.13,
                parse_projected_eigen=False,
            ).defect_entry

            with tempfile.TemporaryDirectory() as tmpdir:
                self._serialise_outputs_to_tmp_tree(tmpdir)
                dp = DefectsParser(  # fully calculator-agnostic parsing of the serialised outputs
                    output_path=tmpdir,
                    dielectric=9.13,
                    calculator="serialized",
                    json_filename=False,
                )

        assert len(dp.defect_dict) == 1
        entry = next(iter(dp.defect_dict.values()))
        assert entry.charge_state == reference_entry.charge_state == -2
        assert entry.calculation_metadata["calculator"] == "serialized"
        assert np.isclose(entry.sc_entry.energy, reference_entry.sc_entry.energy)
        assert np.isclose(entry.bulk_entry.energy, reference_entry.bulk_entry.energy)
        for key in ("vbm", "cbm", "band_gap"):
            assert np.isclose(entry.calculation_metadata[key], reference_entry.calculation_metadata[key])
        assert entry.corrections  # eFNV correction applied from serialised site potentials
        assert np.isclose(
            sum(entry.corrections.values()), sum(reference_entry.corrections.values()), atol=1e-6
        )
        assert np.isclose(entry.get_ediff(), reference_entry.get_ediff(), atol=1e-6)
        assert entry.degeneracy_factors["spin degeneracy"] == 1


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

    def test_doped_analysis_shim(self):
        """
        ``doped.analysis`` forwards to ``doped.parsing`` (and
        ``doped.thermodynamics``) with a ``DeprecationWarning``.
        """
        import doped.analysis
        import doped.parsing
        import doped.thermodynamics

        with pytest.warns(DeprecationWarning, match="doped.analysis has been renamed to doped.parsing"):
            assert doped.analysis.DefectsParser is doped.parsing.DefectsParser

        with pytest.warns(DeprecationWarning, match="import shallow_dopant_binding_energy from "):
            assert (  # moved to doped.thermodynamics, but still forwarded
                doped.analysis.shallow_dopant_binding_energy
                is doped.thermodynamics.shallow_dopant_binding_energy
            )

        with pytest.raises(AttributeError):
            _ = doped.analysis.definitely_not_a_real_attribute

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
