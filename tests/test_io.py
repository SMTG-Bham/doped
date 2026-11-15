"""
Tests for the ``doped.io`` calculator input/output framework.

This includes ``CalculationOutputs``, calculator dispatch, site-potential
sampling from volumetric outputs, and the deprecation shims for pre-``doped.io``
import paths.
"""

import os
import unittest
import warnings

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.core.units import Ry_to_eV, bohr_to_angstrom
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp import VolumetricData
from test_utils import vasp_data_dir

from doped.io import get_calculation_outputs
from doped.io.espresso.outputs import (
    _FALLBACK_SMEARING_WIDTH,
    BOHR_TO_ANGSTROM,
    PAW_COMPENSATION_RADII,
    get_atomic_site_potentials,
)
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
                defect_planar_averaged_potentials=f"{self.defect_dir}/LOCPOT.gz",
                bulk_planar_averaged_potentials=f"{self.bulk_dir}/LOCPOT.gz",
                verbose=False,
            )
            fnv_from_outputs = get_freysoldt_correction(
                defect_entry,
                defect_planar_averaged_potentials=defect_outputs,
                bulk_planar_averaged_potentials=bulk_outputs,
                verbose=False,
            )
            assert np.isclose(fnv_from_files.correction_energy, fnv_from_outputs.correction_energy)
            # TODO: Also hard-test the value here

            efnv_from_files = get_kumagai_correction(
                defect_entry,
                defect_site_potentials=f"{self.defect_dir}/OUTCAR.gz",
                bulk_site_potentials=f"{self.bulk_dir}/OUTCAR.gz",
                verbose=False,
            )
            efnv_from_outputs = get_kumagai_correction(
                defect_entry,
                defect_site_potentials=defect_outputs,
                bulk_site_potentials=bulk_outputs,
                verbose=False,
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


class SitePotentialSmearingTestCase(unittest.TestCase):
    """
    Tests for the Gaussian-smeared atomic-site potentials sampled from
    volumetric potential outputs (``pp.x`` ``.cube``, VASP ``LOCPOT``), used
    for the eFNV finite-size charge correction.
    """

    def test_smearing_matches_analytic_plane_wave_on_odd_grid(self):
        """
        Smearing a single plane wave must scale it by exactly ``exp(-beta^2
        |G|^2 / 2)``, on odd FFT grids as well as even ones.

        ``n = 35`` with ``h = 17`` places the wave in the Nyquist-most
        frequency bin, which is the one (and only) bin where an
        ``arange(-n//2, n//2)`` frequency ladder assigns the wrong Miller
        index for odd ``n`` (``-18`` rather than ``+17``, i.e. ``|G|^2`` 12%
        too large), giving a site potential ~23% too small here.
        """
        n, h, a, beta = 35, 17, 10.0, 0.3  # odd grid; Nyquist-most bin; cubic cell (Å); beta (Å)
        structure = Structure(Lattice.cubic(a), ["H"], [[0.0, 0.0, 0.0]])
        wave = np.cos(2 * np.pi * h * np.arange(n) / n)  # varies along the first lattice vector only
        data = np.broadcast_to(wave[:, None, None], (n, n, n)).copy()

        volumetric_data = VolumetricData(structure, {"total": data})
        # the sampled potentials are negated, for the ``doped``/``pydefect`` eFNV sign convention:
        site_potential = get_atomic_site_potentials(volumetric_data, beta=beta)["site_potentials"][0]

        # exact to machine precision, as this site lies on a grid point; the wrong Miller index
        # would shift the result by ~23%:
        assert np.isclose(site_potential, -np.exp(-0.5 * beta**2 * (2 * np.pi * h / a) ** 2), rtol=1e-12)

    def test_even_grid_frequencies_unchanged(self):
        """
        The previous ``arange``-based frequency ladder was only wrong for odd
        grid dimensions, so no even-grid (e.g. VASP ``LOCPOT``) potentials or
        reference values move with the switch to ``np.fft.fftfreq``.
        """
        for n in (2, 8, 40, 96):
            assert np.array_equal(
                np.fft.fftfreq(n, 1 / n), np.roll(np.arange(-n // 2, n // 2, 1, dtype=int), n // 2)
            )

    def test_per_element_smearing_widths(self):
        """
        With ``beta=None`` (the default), each site must be sampled from the
        grid smeared with *its own* element's width (``rc``, from
        ``PAW_COMPENSATION_RADII``), with elements missing from that table
        falling back to the median tabulated radius with a warning.
        """
        assert np.isclose(_FALLBACK_SMEARING_WIDTH, np.median(list(PAW_COMPENSATION_RADII.values())))
        assert 0.6 < _FALLBACK_SMEARING_WIDTH < 0.7  # bohr; within 25% of the heavy elements it serves
        a, n = 8.0, 20
        structure = Structure(
            Lattice.cubic(a),
            ["Cd", "Te", "Cd", "Tc"],  # Tc is not in ``PAW_COMPENSATION_RADII``
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.5, 0.75], [0.75, 0.25, 0.5]],
        )
        axis = np.arange(n) / n
        data = (  # deterministic, non-separable, with enough structure to resolve the widths apart
            np.sin(2 * np.pi * axis)[:, None, None]
            * np.cos(6 * np.pi * axis)[None, :, None]
            * np.sin(8 * np.pi * axis)[None, None, :]
            + np.cos(4 * np.pi * (axis[:, None, None] + axis[None, :, None] + axis[None, None, :]))
        )
        volumetric_data = VolumetricData(structure, {"total": data})

        with pytest.warns(
            UserWarning, match=r"No PAW compensation-charge radius is tabulated for \['Tc'\]"
        ):
            per_element = get_atomic_site_potentials(volumetric_data)["site_potentials"]

        # widths are tabulated in bohr, and so converted here for this (eV, Å) ``VolumetricData`` input:
        for width_bohr, sites in [
            (PAW_COMPENSATION_RADII["Cd"], [0, 2]),
            (PAW_COMPENSATION_RADII["Te"], [1]),
            (_FALLBACK_SMEARING_WIDTH, [3]),  # the fallback width (table median), for Tc
        ]:
            single = get_atomic_site_potentials(volumetric_data, beta=width_bohr * BOHR_TO_ANGSTROM)[
                "site_potentials"
            ]
            assert np.allclose(per_element[sites], single[sites], rtol=0, atol=1e-12)
            others = [i for i in range(len(structure)) if i not in sites]
            assert not np.any(np.isclose(per_element[others], single[others], rtol=0, atol=1e-12))

    @staticmethod
    def _cube_file(path, structure, data):
        """
        Write ``structure``/``data`` as a Gaussian ``.cube`` file (which uses
        bohr throughout), for testing the ``.cube`` parsing branch.
        """
        matrix = structure.lattice.matrix / bohr_to_angstrom
        lines = ["test cube", "written by doped's test suite", f"{len(structure)} 0.0 0.0 0.0"]
        lines += [
            f"{n} " + " ".join(f"{x:.12f}" for x in vector / n)
            for n, vector in zip(data.shape, matrix, strict=True)
        ]
        lines += [
            f"{site.specie.Z} 0.0 " + " ".join(f"{x:.12f}" for x in site.coords / bohr_to_angstrom)
            for site in structure
        ]
        flat = data.ravel()  # z fastest, as ``.cube`` requires; 6 values per line by convention
        lines += [" ".join(f"{v:.12e}" for v in flat[i : i + 6]) for i in range(0, flat.size, 6)]
        with open(path, "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")

    def test_beta_units_and_dict_input(self):
        """
        ``beta`` is in Å for every input type (``.cube`` potentials are in Ry
        and their geometry in bohr, but ``beta`` is not), and can be given as a
        ``{element symbol: width}`` dict, with unlisted elements taking their
        tabulated default.
        """
        import tempfile

        a, n = 8.0, 8
        structure = Structure(
            Lattice.cubic(a), ["Cd", "Te", "Cd"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.5, 0.75]]
        )
        axis = np.arange(n) / n
        data = np.cos(2 * np.pi * (axis[:, None, None] + 2 * axis[None, :, None] - axis[None, None, :]))
        volumetric_data = VolumetricData(structure, {"total": data})

        with tempfile.TemporaryDirectory() as tmpdir:  # same beta must mean the same width for both
            cube_path = os.path.join(tmpdir, "potential.cube")
            self._cube_file(cube_path, structure, data)
            cube = get_atomic_site_potentials(cube_path, beta=0.3)["site_potentials"]
        locpot_like = get_atomic_site_potentials(volumetric_data, beta=0.3)["site_potentials"]
        assert np.allclose(cube, Ry_to_eV * locpot_like, rtol=1e-8)  # only the potential units differ

        # a dict sets the width per element, with any not given taking the tabulated default:
        default = PAW_COMPENSATION_RADII["Te"] * BOHR_TO_ANGSTROM
        mixed = get_atomic_site_potentials(volumetric_data, beta={"Cd": 0.4})["site_potentials"]
        for width, sites in [(0.4, [0, 2]), (default, [1])]:
            single = get_atomic_site_potentials(volumetric_data, beta=width)["site_potentials"]
            assert np.allclose(mixed[sites], single[sites], rtol=0, atol=1e-12)
            others = [i for i in range(len(structure)) if i not in sites]
            assert not np.any(np.isclose(mixed[others], single[others], rtol=0, atol=1e-12))

        # keys for elements absent from the structure are ignored without warning, as the same dict
        # is used for the bulk and every defect supercell:
        assert np.array_equal(
            mixed,
            get_atomic_site_potentials(volumetric_data, beta={"Cd": 0.4, "Mg": 0.3})["site_potentials"],
        )
