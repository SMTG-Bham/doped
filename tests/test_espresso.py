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
#  - DOS: ``examples/MgO_qe/DOS`` is added here but nothing exercises it yet (``FermiSolver`` with
#    espresso; see the tracking issue). ``examples/MgO_qe/Competing_phases`` is parsed in
#    ``EspressoCompetingPhasesTestCase`` below.

import os
import re
import shutil
import tempfile
import unittest
import warnings

import numpy as np
import pytest
from monty.serialization import loadfn
from pymatgen.core import Lattice, Structure
from pymatgen.core.units import Ry_to_eV, bohr_to_angstrom
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp import VolumetricData

# the espresso backend needs the optional ``pymatgen-io-espresso`` dependency; skip (rather than fail
# collection of) this module without it:
pytest.importorskip("pymatgen.io.espresso")

from doped.chemical_potentials import CompetingPhasesAnalyzer
from doped.io import get_calculation_outputs
from doped.io.espresso.inputs import (
    DefectRelaxSetQE,
    DopedDictSetQE,
    _competing_phase_system_settings,
    _get_atomic_species,
)
from doped.io.espresso.outputs import (
    _FALLBACK_SMEARING_WIDTH,
    BOHR_TO_ANGSTROM,
    MISMATCH_WARNING_SPECS,
    PAW_COMPENSATION_RADII,
    check_run_compatibility,
    get_atomic_site_potentials,
    get_competing_phase_entry,
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


def _parse_pw_in(path: str) -> dict:
    """
    Parse the ``&SYSTEM`` settings, ``ATOMIC_SPECIES`` labels, site labels and
    ``K_POINTS`` card of a written ``pw.in``, asserting that it satisfies the
    ``pw.x`` input rules (from ``PW/src/input.f90``) for spin polarisation,
    k-points and atomic species.
    """
    with open(path) as f:
        text = f.read()

    def group(pattern: str, flags: int = 0) -> str:
        match = re.search(pattern, text, flags)
        assert match is not None, pattern
        return match.group(1)

    system_text = group(r"&SYSTEM(.*?)\n\s*/", re.S | re.I)
    system = dict(re.findall(r"^\s*(\S+)\s*=\s*(\S+)", system_text, re.M))
    species = group(r"ATOMIC_SPECIES.*?\n((?:[ \t]*\S+[ \t]+[\d.]+[ \t]+\S+\n)+)")
    species_labels = [line.split()[0] for line in species.splitlines()]
    positions = group(r"ATOMIC_POSITIONS.*?\n((?:[ \t]*\S+(?:[ \t]+-?[\d.]+){3}\n)+)")
    site_labels = [line.split()[0] for line in positions.splitlines()]
    k_points = group(r"(K_POINTS.*\n.*)")

    assert int(system["ntyp"]) == len(species_labels)
    assert all(len(label) <= 3 for label in species_labels)
    assert set(site_labels) == set(species_labels)
    seeds = [float(v) for k, v in system.items() if k.startswith("starting_magnetization(")]
    if system.get("noncolin") == ".true.":  # "gamma_only and noncolin not allowed"
        assert "nspin" not in system
        assert "tot_magnetization" not in system
        assert "gamma" not in k_points
    if "tot_magnetization" in system:  # "tot_magnetization requires nspin=2"
        assert system.get("nspin") == "2"
    if system.get("nspin") == "2":
        if system.get("occupations", "'fixed'") == "'fixed'":  # "fixed occupations and lsda need..."
            assert float(system["tot_magnetization"]).is_integer()
        elif "tot_magnetization" not in system:  # some starting_magnetization must be set
            assert any(seeds)

    return {"system": system, "species": species_labels, "sites": site_labels, "k_points": k_points}


class EspressoInputsTestCase(unittest.TestCase):
    """
    Writing espresso ``pw.in`` files: spin polarisation (with atomic species
    split by magnetic moment), ``K_POINTS`` cards and defect calculation
    defaults.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir)
        self.afm = Structure(
            Lattice.cubic(4.3),
            ["Fe", "Fe", "O", "O"],
            [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0], [0, 0.5, 0]],
        )
        self.afm.add_site_property("magmom", [4.0, -4.0, 0.0, 0.0])
        self.fm = self.afm.copy(site_properties={"magmom": [4.0, 4.0, 0.0, 0.0]})

    def _write(self, dict_set: DopedDictSetQE, name: str = "pw") -> dict:
        with warnings.catch_warnings():  # missing pseudopotential files warning not under test here
            warnings.simplefilter("ignore")
            dict_set.write_input(os.path.join(self.tmpdir, name))
        return _parse_pw_in(os.path.join(self.tmpdir, name, "pw.in"))

    def _write_competing_phase(
        self, structure, magnetization, is_metal=False, soc=False, kpoints=(2, 2, 2)
    ):
        entry = ComputedStructureEntry(
            structure, -1, data={"summary": {"total_magnetization": magnetization}}
        )
        system_settings = _competing_phase_system_settings(entry, {}, soc, structure, is_metal)
        return self._write(
            DopedDictSetQE(
                structure,
                kpoints=list(kpoints) if kpoints else None,
                is_metal=is_metal,
                soc=soc,
                user_system_settings=system_settings,
            )
        )

    def test_competing_phase_spin_settings(self):
        afm = self._write_competing_phase(self.afm, 0)  # insulator, so fixed occupations
        assert afm["system"]["tot_magnetization"] == "0"
        assert afm["species"] == ["O", "Fe", "Fe1"]  # Fe split into species by moment sign
        assert afm["sites"] == ["Fe", "Fe1", "O", "O"]
        assert [afm["system"][f"starting_magnetization({i})"] for i in (1, 2, 3)] == ["0.0", "4.0", "-4.0"]

        fm = self._write_competing_phase(self.fm, 8)
        assert fm["system"]["tot_magnetization"] == "8"
        assert fm["species"] == ["O", "Fe"]  # no seeds needed with a fixed (non-negative) magnetization
        assert not any(key.startswith("starting_magnetization") for key in fm["system"])

        # MP magnetization inconsistent with the site moments; sum of site moments used:
        assert self._write_competing_phase(self.fm, 0)["system"]["tot_magnetization"] == "8"

        afm_metal = self._write_competing_phase(self.afm, 0, is_metal=True)
        assert "tot_magnetization" not in afm_metal["system"]  # free magnetization with smearing
        assert afm_metal["system"]["occupations"] == "'smearing'"
        assert afm_metal["species"] == ["O", "Fe", "Fe1"]

        fm_metal = self._write_competing_phase(
            self.afm.copy(site_properties={"magmom": [0] * 4}), 2.4, True
        )
        assert fm_metal["species"] == ["O", "Fe"]  # no site moments; mean moment seeded on all species
        assert fm_metal["system"]["starting_magnetization(1)"] == "0.6"

        afm_soc = self._write_competing_phase(self.afm, 0, soc=True, kpoints=None)
        assert afm_soc["system"]["noncolin"] == ".true."
        assert afm_soc["species"] == ["O", "Fe", "Fe1"]
        assert afm_soc["k_points"].split() == ["K_POINTS", "{automatic}", "1", "1", "1", "0", "0", "0"]

        non_magnetic = self._write_competing_phase(self.afm.copy(site_properties={"magmom": [0] * 4}), 0)
        assert "nspin" not in non_magnetic["system"]
        assert non_magnetic["species"] == ["O", "Fe"]

    def test_species_split_by_moment(self):
        ligand = self.afm.copy(site_properties={"magmom": [4.0, -4.0, 0.3, -0.3]})
        written = self._write_competing_phase(ligand, 0)
        assert written["species"] == ["O", "O1", "Fe", "Fe1"]
        # raw moments in μB as seeds (QE >= 7.3 interprets all as μB when any is >= 1):
        assert [written["system"][f"starting_magnetization({i})"] for i in (1, 2, 3, 4)] == [
            "0.3",
            "-0.3",
            "4.0",
            "-4.0",
        ]

        # moments of the same sign within 0.5 μB are grouped:
        labels, species = _get_atomic_species(self.afm, magmoms=[4.0, 3.6, 0.0, 0.0])
        assert labels == ["Fe", "Fe", "O", "O"]
        assert species == [("O", "O", 0.0), ("Fe", "Fe", 4.0)]
        chain = Structure(Lattice.cubic(30), ["Fe"] * 11, [[i / 11, 0, 0] for i in range(11)])
        with pytest.raises(ValueError, match="at most 3 characters"):
            _get_atomic_species(chain, magmoms=list(range(1, 12)))  # 11 distinct moments -> "Fe10" label

    def test_gamma_only_kpoints(self):
        o2 = Structure(Lattice.cubic(15), ["O", "O"], [[0, 0, 0], [0.08, 0, 0]])
        written = self._write_competing_phase(o2, 2, kpoints=None)  # triplet O2 molecule
        assert written["k_points"].split()[:2] == ["K_POINTS", "{gamma}"]
        assert written["system"]["tot_magnetization"] == "2"

    def test_defect_relax_set_defaults(self):
        bulk = (
            Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.21), ["Mg", "O"], [[0] * 3, [0.5] * 3]) * 2
        )
        defect = bulk.copy()
        defect.remove_sites([len(defect) - 1])  # v_O; odd number of electrons in the +1 charge state
        for soc in [False, True]:
            relax_set = DefectRelaxSetQE(defect, charge_state=1, soc=soc, bulk_supercell=bulk)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                written = relax_set.write_all(
                    os.path.join(self.tmpdir, f"v_O_{soc}"), gamma=True, bulk=True
                )
            parsed = {
                (stage, kind): _parse_pw_in(path)
                for stage, paths in written.items()
                for kind, path in paths.items()
            }
            assert ("espresso_ncl", "defect") in parsed if soc else ("espresso_std", "defect") in parsed
            for (stage, kind), pw_in in parsed.items():
                system = pw_in["system"]
                assert system["occupations"] == "'smearing'"
                assert float(system["degauss"]) == pytest.approx(0.05 / Ry_to_eV, abs=1e-5)
                if stage == "espresso_ncl":
                    assert system["noncolin"] == ".true."
                else:  # only ``espresso_ncl`` is SOC; collinear, with the spin state set by the parity
                    assert "noncolin" not in system
                    assert system["tot_magnetization"] == ("0" if kind == "bulk" else "1")
            assert parsed["espresso_gamma", "defect"]["k_points"].split()[:2] == ["K_POINTS", "{gamma}"]


class EspressoOutputsLabelsTestCase(unittest.TestCase):
    """
    Numbered espresso species labels (e.g. ``Mg``/``Mg1``) are parsed as
    elements.
    """

    def test_numbered_species_labels(self):
        src = os.path.join(
            os.path.dirname(__file__),
            "../examples/MgO_qe/Competing_phases/MgO_Fm-3m_EaH_0/espresso_std/espresso.xml",
        )
        with open(src) as f:
            text = (
                f.read()
                .replace('<atom name="Mg"', '<atom name="Mg1"')
                .replace('<species name="Mg"', '<species name="Mg1"')
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "espresso.xml")
            with open(path, "w") as f:
                f.write(text)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                run = get_espresso_run(path)
                entry = get_competing_phase_entry(path)
        assert not any("Species labels contain numbers" in str(warning.message) for warning in w)
        assert run.atomic_symbols == ["Mg", "O"]
        for structure in [
            run.initial_structure,
            run.final_structure,
            *(s["structure"] for s in run.ionic_steps),
        ]:
            assert {str(species) for species in structure.species} == {"Mg", "O"}
        assert entry.data["elements"] == ["Mg", "O"]
        assert entry.composition.reduced_formula == "MgO"


class EspressoCompetingPhasesTestCase(unittest.TestCase):
    """
    Parsing the MgO espresso competing phase calculations.
    """

    def test_mgo_competing_phases(self):
        base = os.path.join(os.path.dirname(__file__), "../examples/MgO_qe/Competing_phases")
        cpa = CompetingPhasesAnalyzer("MgO", base, calculator="espresso")
        assert sorted(entry.name for entry in cpa.entries) == ["Mg", "Mg", "MgO", "O2"]  # hcp & dhcp Mg
        assert cpa.chempots == loadfn(os.path.join(base, "MgO_chempots_QE.json"))
