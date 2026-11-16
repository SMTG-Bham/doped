"""
Tests for Quantum ESPRESSO (``pw.x``/``pp.x``) defect parsing and charge
corrections, using the MgO example data in ``examples/MgO_qe``.
"""

# TODO: This covers the correction values only, at present. The remaining espresso tests live in
#  ``tests/test_quantum_espresso.py`` and ``tests/test_quantum_espresso_anisotropic.py`` on the
#  ``atharva_doped_QE`` branch and still need porting. Note that file is stale (last updated
#  2026-04-17, while ``examples/MgO_qe`` was reorganised into ``Defects/``/``Unrelaxed/`` subfolders on
#  2026-06-10 in 3bc32eaf, so its paths no longer resolve) and has the anisotropic test file
#  concatenated into it from line 845 onwards, so porting is not a copy-paste job. Also note the API
#  drift: ``test_analysis`` -> ``test_parsing``, ``RunParser`` is gone (use
#  ``doped.io.espresso.outputs.get_atomic_site_potentials``), ``code=`` -> ``calculator=``, ``beta`` is
#  now in Å rather than bohr (so ``beta=3`` -> ``beta=3 * BOHR_TO_ANGSTROM``), and ``pp_folder`` is no
#  longer a ``DefectsParser`` kwarg. Still to port, with the data each needs:
#  - ``Mg_O_Unrelaxed_+1``: its correction, and the comparison against VASP at matching ENCUT (this is
#    the tightest QE/VASP check, as the ionic positions are identical). Needs
#    ``examples/MgO_qe/Unrelaxed`` (already added here) and ``examples/MgO/ENCUT_400_Defects`` (90 MB,
#    already gzipped). Note it sits in a sibling folder of ``Defects/``, so it needs its own parse --
#    the stored ``MgO_defect_dict.json`` still has it as a 5th entry, from the old flat layout.
#  - ``DefectsParser`` behaviour: the projected-magnetisation warning, ``check_DefectsParser``,
#    ``skip_corrections``, the no-dielectric warning, the band-gap error, serial (no multiprocessing)
#    parsing, and the JSON round-trip. Needs nothing new.
#  - Re-deriving corrections from a loaded entry with ``get_kumagai_correction``, and the
#    correction-error (``kumagai_charge_correction_error``) checks. Needs nothing new.
#  - ``sxdefectalign`` comparisons: doped vs ``sxdefectalign`` site potentials for q = +1 to +3 and for
#    the unrelaxed case, at the default and at ``beta = 3`` bohr, plus QE-vs-VASP of both. Needs
#    ``examples/MgO_qe/sxdefectalign`` (already added here).
#  - Correction plots: 10 ``mpl_image_compare`` tests (side-by-side QE vs VASP eFNV plots, and doped vs
#    ``sxdefectalign`` potential plots). Needs the 10 ``QE_*``/``SST_*`` baselines in
#    ``tests/data/remote_baseline_plots``, which will need regenerating anyway.
#  - The whole anisotropic Sb2Si2Te6 set (``v_Sb_-3``, with a tensor dielectric, and with/without SOC):
#    the same four groups again. Needs ``examples/Sb2Si2Te6_qe`` (194 MB raw, ~50 MB gzipped); note
#    three ``examples/Sb2Si2Te6/*.json`` files differ between that branch and this one.
#  - Competing phases and DOS: ``examples/MgO_qe/Competing_phases`` and ``examples/MgO_qe/DOS`` are
#    added here but nothing exercises them yet; a ``CompetingPhasesAnalyzer``/``FermiSolver`` backend
#    for espresso is separate work (see the tracking issue).

import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.core.units import Ry_to_eV, bohr_to_angstrom
from pymatgen.io.vasp import VolumetricData

# the espresso backend needs the optional ``pymatgen-io-espresso`` dependency; skip (rather than fail
# collection of) this module without it:
pytest.importorskip("pymatgen.io.espresso")

from doped.io import get_calculation_outputs
from doped.io.espresso.outputs import (
    _FALLBACK_SMEARING_WIDTH,
    BOHR_TO_ANGSTROM,
    MISMATCH_WARNING_SPECS,
    PAW_COMPENSATION_RADII,
    check_run_compatibility,
    get_atomic_site_potentials,
    get_espresso_run,
)
from doped.io.utils import _warn_calculation_mismatches
from doped.parsing import DefectsParser

MGO_DIELECTRIC = 8.8963

# Kumagai (eFNV) corrections for the same Mg_O defects from VASP, parsed from
# ``examples/MgO/Defects/Pre_Calculated_Results`` with the dielectric constant above:
VASP_CORRECTIONS = {"Mg_O_+1": 0.1988, "Mg_O_+2": 0.7233, "Mg_O_+3": 1.5717, "Mg_O_+4": 2.5860}


def _parse_mgo_qe(**kwargs) -> dict:
    """
    Parse the MgO espresso defect calculations, returning ``{name: kumagai
    correction (eV)}``.
    """
    defects_dir = os.path.join(os.path.dirname(__file__), "../examples/MgO_qe/Defects")
    with warnings.catch_warnings():  # projected-magnetisation warning is not under test here
        warnings.simplefilter("ignore")
        defect_dict = DefectsParser(
            output_path=defects_dir,
            bulk_path=os.path.join(defects_dir, "MgO_bulk"),
            calculator="espresso",
            dielectric=MGO_DIELECTRIC,
            json_filename=False,
            **kwargs,
        ).defect_dict

    return {name: entry.corrections["kumagai_charge_correction"] for name, entry in defect_dict.items()}


class EspressoMgOCorrectionsTestCase(unittest.TestCase):
    """
    Kumagai (eFNV) corrections for Mg_O in MgO, parsed from espresso outputs.
    """

    @classmethod
    def setUpClass(cls):
        cls.corrections = _parse_mgo_qe()  # default (per-element) smearing widths

    def test_parsed_defects(self):
        assert set(self.corrections) == set(VASP_CORRECTIONS)

    def test_kumagai_corrections(self):
        expected = {"Mg_O_+1": 0.1973, "Mg_O_+2": 0.7253, "Mg_O_+3": 1.5760, "Mg_O_+4": 2.5951}
        for name, value in expected.items():
            actual = self.corrections[name]
            assert np.isclose(actual, value, atol=2e-3), f"{name}: {actual}"

    def test_agreement_with_vasp(self):
        """
        The espresso and VASP corrections are for the same defects in the same
        material, so they should agree; this is the only check that the smeared
        site potentials are right which does not go through the code that
        produces them.
        """
        diffs = {name: abs(self.corrections[name] - ref) for name, ref in VASP_CORRECTIONS.items()}
        assert np.mean(list(diffs.values())) < 0.006, diffs
        assert max(diffs.values()) < 0.011, diffs

    def test_beta_dict_matches_the_default(self):
        """
        Setting the widths explicitly to the tabulated radii, as a dict, must
        reproduce the default parsing exactly.
        """
        widths = {el: PAW_COMPENSATION_RADII[el] * BOHR_TO_ANGSTROM for el in ("Mg", "O")}
        assert _parse_mgo_qe(beta=widths) == self.corrections


class EspressoBetaTestCase(unittest.TestCase):
    """
    The Gaussian smearing width, ``beta``, which defaults to each element's PAW
    compensation-charge radius.
    """

    def test_explicit_global_beta(self):
        """
        A single (too large) ``beta`` for every element, in Å, gives worse
        agreement with VASP than the per-element default.
        """
        corrections = _parse_mgo_qe(beta=1.2 * BOHR_TO_ANGSTROM)
        expected = {"Mg_O_+1": 0.1609, "Mg_O_+2": 0.6566, "Mg_O_+3": 1.4714, "Mg_O_+4": 2.4662}
        for name, value in expected.items():
            assert np.isclose(corrections[name], value, atol=2e-3), f"{name}: {corrections[name]}"

        diffs = [abs(corrections[name] - ref) for name, ref in VASP_CORRECTIONS.items()]
        assert np.mean(diffs) > 0.05, diffs  # cf. <0.006 eV for the per-element default


class EspressoCalculationOutputsTestCase(unittest.TestCase):
    """
    ``get_calculation_outputs(..., calculator="espresso")`` fields.
    """

    def test_charge_from_xml(self):
        """
        The supercell charge is read from the ``pw.x`` XML (``tot_charge``),
        including with ``pymatgen-io-espresso`` versions whose ``PWxml`` has no
        ``total_charge`` attribute, rather than being left to the folder-name
        guess -- so it is tested here in folders without a ``_±q`` suffix.
        """
        defects_dir = os.path.join(os.path.dirname(__file__), "../examples/MgO_qe/Defects")
        for name, charge in [("Mg_O_+1", 1), ("Mg_O_+4", 4), ("MgO_bulk", 0)]:
            with tempfile.TemporaryDirectory() as tmpdir:
                shutil.copy(os.path.join(defects_dir, name, "espresso_std", "espresso.xml"), tmpdir)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # pymatgen-io-espresso ``NotImplementedForQE`` etc.
                    outputs = get_calculation_outputs(tmpdir, calculator="espresso")
            assert outputs.charge == charge, f"{name}: {outputs.charge}"

    def test_smearing_not_compared_between_defect_and_bulk(self):
        """
        Smearing settings (``occupations``/``smearing``/``degauss``) can
        legitimately differ between defect and bulk supercell calculations, so
        are not flagged as mismatches -- e.g. the smeared ``Mg_O_+1`` vs fixed-
        occupation bulk MgO calculations.

        Also checks the run metadata serialises without warnings
        (``PWxml.as_dict()`` needs attributes that some
        ``pymatgen-io-espresso`` versions never set, which ``get_espresso_run``
        defaults).
        """
        defects_dir = os.path.join(os.path.dirname(__file__), "../examples/MgO_qe/Defects")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pymatgen-io-espresso ``NotImplementedForQE`` etc.
            defect_pwxml, bulk_pwxml = (
                get_espresso_run(
                    os.path.join(defects_dir, name, "espresso_std", "espresso.xml"),
                    parse_projected_eigen=False,
                )
                for name in ("Mg_O_+1", "MgO_bulk")
            )
        defect_occupations, bulk_occupations = (
            pwxml.parameters["bands"]["occupations"] for pwxml in (defect_pwxml, bulk_pwxml)
        )
        assert defect_occupations != bulk_occupations  # smeared vs fixed occupations
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compatibility = check_run_compatibility(defect_pwxml, bulk_pwxml)
        assert compatibility["mismatching_QE_input_params"] is False
        assert not [x for x in w if "mismatch" in str(x.message).lower()]
        assert not [x for x in w if "could not be serialised" in str(x.message)]
        assert "defect_vasprun_dict" in compatibility["run_metadata"]

    def test_mismatch_warning_specs(self):
        """
        The espresso ``MISMATCH_WARNING_SPECS`` render with the calculator-
        agnostic mismatch-warning machinery.
        """
        metadata = {
            "mismatching_QE_input_params": [("ecutwfc", 60.0, 50.0)],
            "mismatching_pseudo_filenames": [("O.pbe-n-kjpaw_psl.0.1.UPF", "O.pbe-rrkjus.UPF")],
            "mismatching_kpoints": [((2, 2, 2), (3, 3, 3))],
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_calculation_mismatches([("Mg_O_+1", metadata)], MISMATCH_WARNING_SPECS)
        messages = [str(x.message) for x in w]
        assert len(messages) == 3
        assert "('ecutwfc', 60.0, 50.0)" in messages[0]
        assert "value in defect calculation, value in bulk calculation" in messages[0]
        assert (
            "(pseudopotential in defect calculation, pseudopotential in bulk calculation)" in messages[1]
        )
        assert "Mg_O_+1: " in messages[1]
        assert "O.pbe-rrkjus.UPF" in messages[1]
        assert "(defect k-points, bulk k-points)" in messages[2]


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

        # exact; as this site lies on a grid point; the wrong Miller index would shift the result by ~23%:
        assert np.isclose(site_potential, -np.exp(-0.5 * beta**2 * (2 * np.pi * h / a) ** 2), rtol=1e-12)

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
