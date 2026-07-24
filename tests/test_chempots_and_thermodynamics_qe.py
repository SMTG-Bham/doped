"""
Tests for the Quantum ESPRESSO (``pw.x``) competing-phase / chemical-potential
functionality in ``doped.chemical_potentials``:

TODO: QE code yet to be tested for extrinsic defects

"""

import os
import shutil
import tempfile
import unittest
import warnings
from copy import deepcopy

import matplotlib as mpl
import numpy as np
import pytest
from monty.serialization import loadfn
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core.composition import Composition
from pymatgen.electronic_structure.dos import FermiDos
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.io.espresso.inputs.pwin import PWin
from test_utils import EXAMPLE_DIR, _run_heavy_tests, api_key

from doped.chemical_potentials import CompetingPhasesAnalyzerQE, CompetingPhasesQE
from doped.qe import qe_SSSP_pseudo_filenames
from doped.thermodynamics import DefectThermodynamics, get_fermi_dos_from_espresso_dos

mpl.use("Agg")  # don't show interactive plots if testing from CLI locally


COMPETING_PHASES_QE_DIR = os.path.join(EXAMPLE_DIR, "MgO_qe", "CompetingPhases_MgO_QE")
# QE competing-phase *outputs* (espresso.xml) + saved chempots for MgO:
MGO_QE_CP_DIR = os.path.join(EXAMPLE_DIR, "MgO_qe", "Competing_phases_worked_examples")
MGO_SAVED_CHEMPOTS = os.path.join(MGO_QE_CP_DIR, "MgO_chempots_QE.json")

# the five MgO competing phases (folder names) generated for the example:
MOLECULE_PHASE = "O2_mmm_EaH_0"
NON_MOLECULE_PHASES = {
    "MgO_Fm-3m_EaH_0",
    "Mg_Fm-3m_EaH_0",
    "Mg_P6_3mmc_EaH_0.009",
    "Mg_R-3m_EaH_0.003",
}
EXPECTED_PHASES = NON_MOLECULE_PHASES | {MOLECULE_PHASE}
# default ``ecut_convergence=(20, 90, 10)`` sweep (max inclusive):
EXPECTED_ECUTS = [20, 30, 40, 50, 60, 70, 80, 90]
# default host-named relative path written by the QE setup methods
# ("{host}_QE/CompetingPhases_{host}_QE"):
DEFAULT_QE_CP_RELPATH = os.path.join("MgO_QE", "CompetingPhases_MgO_QE")

# ── chemical-potential values calculated in the notebook (Also saved to MgO_chempots_QE.json) ───
MGO_ELEMENTAL_REFS = {"Mg": -457.3966, "O": -564.96127}  # eV/atom
# MgO formation energy = chempot drop at the opposite-rich limit (eV/f.u.):
MGO_FORMATION_ENERGY = -5.38616


def _kpoint_grid_from_folder_name(kname: str) -> list[int]:
    """``"k_10,10,10" -> [10, 10, 10]`` (strip leading ``k``/``_`` padding)."""
    return [int(k) for k in kname.lstrip("k_").split(",")]


def _canonicalise_chempot_dict(chempots: dict) -> dict:
    """
    Copy of a chempot dict with limit keys canonicalised (bordering phases
    sorted alphabetically), so two dicts can be compared regardless of the
    order in which the ``PhaseDiagram`` enumerated phases at each facet.
    """
    out: dict = {}
    for outer_k, outer_v in chempots.items():
        if isinstance(outer_v, dict):
            out[outer_k] = {
                ("-".join(sorted(k.split("-"))) if isinstance(k, str) and "-" in k else k): v
                for k, v in outer_v.items()
            }
        else:
            out[outer_k] = outer_v
    return out


def _compare_chempot_dicts(dict1: dict, dict2: dict):
    """Recursively assert two (canonicalised) chempot dicts are ~equal."""
    assert set(dict1) == set(dict2)
    for key, val in dict1.items():
        if isinstance(val, dict):
            _compare_chempot_dicts(val, dict2[key])
        else:
            assert np.isclose(val, dict2[key], atol=1e-5), f"{key}: {val} != {dict2[key]}"


def _assert_valid_pw_input(pw: PWin):
    """Common structural checks for any generated competing-phase ``pw.in``."""
    # all structures are written with an explicit lattice (ibrav = 0):
    assert pw.system["ibrav"] == 0
    # required namelists / cards present:
    for nl in ("control", "system", "electrons"):
        assert pw.namelists[nl] is not None, f"&{nl.upper()} missing"
    for card in ("atomic_species", "atomic_positions", "k_points", "cell_parameters"):
        assert pw.cards[card] is not None, f"{card} card missing"
    assert "calculation" in pw.control
    for key in ("ecutwfc", "nat", "ntyp", "ecutrho", "ibrav"):
        assert key in pw.system
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert pw.validate() is True

#TODO: Mention that SSSP pseudopotentials must be downloaded by the user.
def _assert_sssp_pseudos(pw: PWin):
    """ATOMIC_SPECIES uses the bundled SSSP 1.3.0 PBE Efficiency filenames."""
    files = dict(zip(pw.atomic_species.symbols, pw.atomic_species.files))
    for sym, fname in files.items():
        assert fname == qe_SSSP_pseudo_filenames[sym], f"{sym}: {fname}"


def _phase_is_metal(phase: str) -> bool:
    """
    Whether a competing-phase folder is metallic, inferred from its composition
    (an elemental phase of a metallic element).
    """
    comp = Composition(phase.split("_")[0])
    return len(comp.elements) == 1 and comp.elements[0].is_metal


def _assert_smearing(pw: PWin, phase: str):
    """
    Assert the ``&SYSTEM`` smearing is appropriate for ``phase``: metallic
    phases carry the default Gaussian smearing, while non-metals (and
    molecules) have none (QE's default ``fixed`` occupations).
    """
    if _phase_is_metal(phase):
        assert pw.system["occupations"] == "smearing"
        assert pw.system["smearing"] == "gaussian"
        assert pw.system["degauss"] == 0.005
    else:
        assert "smearing" not in pw.system
        assert "degauss" not in pw.system



class CompetingPhasesQEExampleFilesTestCase(unittest.TestCase):
    def test_competing_phases_directory_and_phase_folders_exist(self):
        """The example ``CompetingPhases`` dir holds the five MgO phases."""
        assert os.path.isdir(COMPETING_PHASES_QE_DIR), COMPETING_PHASES_QE_DIR
        present = {
            d
            for d in os.listdir(COMPETING_PHASES_QE_DIR)
            if os.path.isdir(os.path.join(COMPETING_PHASES_QE_DIR, d))
        }
        assert EXPECTED_PHASES == present, f"missing phase folders: {EXPECTED_PHASES - present}"

    def test_per_phase_subtree_layout(self):
        """
        Each phase has an ``espresso_std/pw.in`` and an ``ecut_convergence``
        sweep; only the (non-molecule) solids also get a ``kpoint_converge``
        sweep (Γ-only is exact for the O2 molecule-in-a-box).
        """
        for phase in EXPECTED_PHASES:
            phase_dir = os.path.join(COMPETING_PHASES_QE_DIR, phase)

            std_input = os.path.join(phase_dir, "espresso_std", "pw.in")
            assert os.path.isfile(std_input), f"missing {std_input}"

            ecut_dir = os.path.join(phase_dir, "ecut_convergence")
            assert os.path.isdir(ecut_dir), f"missing {ecut_dir}"
            for ecut in EXPECTED_ECUTS:
                assert os.path.isfile(
                    os.path.join(ecut_dir, f"ecutwfc_{ecut}", "pw.in")
                ), f"missing ecutwfc_{ecut} for {phase}"

            found_ecut_folders = {
                d for d in os.listdir(ecut_dir) if os.path.isdir(os.path.join(ecut_dir, d))
            }
            if found_ecut_folders != {f"ecutwfc_{e}" for e in EXPECTED_ECUTS}:
                warnings.warn(
                    f"ecut_convergence folders for {phase} do not match the default "
                    f"(20, 90, 10) sweep: found {sorted(found_ecut_folders)}"
                )

            kpoint_dir = os.path.join(phase_dir, "kpoint_converge")
            if phase == MOLECULE_PHASE:
                assert not os.path.isdir(kpoint_dir), "molecule should have no k-point sweep"
            else:
                assert os.path.isdir(kpoint_dir), f"missing {kpoint_dir}"
                assert os.listdir(kpoint_dir), f"empty k-point sweep for {phase}"

    def test_all_example_inputs_are_valid(self):
        """
        A ``pw.in`` is present at every expected location, and each one parses,
        validates and uses SSSP pseudos.
        """

        def _check_input(phase, *subpath):
            path = os.path.join(COMPETING_PHASES_QE_DIR, phase, *subpath, "pw.in")
            assert os.path.isfile(path), f"missing {path}"
            pw = PWin.from_file(path)
            _assert_valid_pw_input(pw)
            _assert_sssp_pseudos(pw)
            _assert_smearing(pw, phase)  

        for phase in EXPECTED_PHASES:
            _check_input(phase, "espresso_std")  
            for ecut in EXPECTED_ECUTS: 
                _check_input(phase, "ecut_convergence", f"ecutwfc_{ecut}")


        for phase in NON_MOLECULE_PHASES:
            kpoint_dir = os.path.join(COMPETING_PHASES_QE_DIR, phase, "kpoint_converge")
            kgrids = [d for d in os.listdir(kpoint_dir) if os.path.isdir(os.path.join(kpoint_dir, d))]
            assert kgrids, f"empty k-point sweep for {phase}"
            for kname in kgrids:
                _check_input(phase, "kpoint_converge", kname)

        assert not os.path.isdir(  
            os.path.join(COMPETING_PHASES_QE_DIR, MOLECULE_PHASE, "kpoint_converge")
        )

    def test_ecut_convergence_inputs(self):
        """``ecut_convergence/ecutwfc_<N>/pw.in`` is an SCF at ``ecutwfc = N``."""
        for phase in EXPECTED_PHASES:
            ecut_dir = os.path.join(COMPETING_PHASES_QE_DIR, phase, "ecut_convergence")
            for ecut in EXPECTED_ECUTS:
                pw = PWin.from_file(os.path.join(ecut_dir, f"ecutwfc_{ecut}", "pw.in"))
                assert pw.control["calculation"] == "scf", (
                    f"&CONTROL calculation should be 'scf' for the phase: {phase} (ecutwfc_{ecut})"
                )
                assert pw.system["ecutwfc"] == ecut, (
                    f"&SYSTEM ecutwfc should be {ecut} for the phase: {phase}"
                )
                assert pw.system["ecutrho"] == 240, (
                    f"&SYSTEM ecutrho should be 240 for the phase: {phase} (ecutwfc_{ecut})"
                )
                if phase == MOLECULE_PHASE:  # molecules swept with Γ-only sampling
                    assert str(pw.k_points.option) == "gamma", f"Kpoints should have gamma option for the molecule phase: {phase}"

    def test_kpoint_convergence_inputs(self):
        """``kpoint_converge/k.../pw.in`` is an SCF whose grid matches its folder."""
        for phase in NON_MOLECULE_PHASES:
            kpoint_dir = os.path.join(COMPETING_PHASES_QE_DIR, phase, "kpoint_converge")
            for kname in os.listdir(kpoint_dir):
                pw = PWin.from_file(os.path.join(kpoint_dir, kname, "pw.in"))
                assert pw.control["calculation"] == "scf", (
                    f"&CONTROL calculation should be 'scf' for the phase: {phase}/{kname}"
                )
                assert str(pw.k_points.option) == "automatic", (
                    f"K_POINTS option should be 'automatic' for the phase: {phase}/{kname}"
                )
                assert list(pw.k_points.grid) == _kpoint_grid_from_folder_name(kname), (
                    f"K_POINTS grid should match the folder name {kname} for the phase: {phase}"
                )

    def test_std_setup_inputs(self):
        """
        ``espresso_std/pw.in`` is a (full) relaxation for solids and a
        fixed-cell ``relax`` for the O2 molecule (which is also Γ-only and
        spin-polarised as an O2 triplet).
        """
        for phase in NON_MOLECULE_PHASES:
            pw = PWin.from_file(os.path.join(COMPETING_PHASES_QE_DIR, phase, "espresso_std", "pw.in"))
            assert pw.control["calculation"] == "vc-relax", (
                f"&CONTROL calculation should be 'vc-relax' for the phase: {phase}"
            )
            assert str(pw.k_points.option) == "automatic", (
                f"K_POINTS option should be 'automatic' for the phase: {phase}"
            )

        mol = PWin.from_file(
            os.path.join(COMPETING_PHASES_QE_DIR, MOLECULE_PHASE, "espresso_std", "pw.in")
        )
        assert mol.control["calculation"] == "relax", (  # fixed cell for molecule-in-a-box
            f"&CONTROL calculation should be 'relax' for the O2 molecule phase: {MOLECULE_PHASE}"
        )
        assert str(mol.k_points.option) == "gamma", (
            f"K_POINTS option should be 'gamma' for the O2 molecule phase: {MOLECULE_PHASE}"
        )
        assert mol.system["nspin"] == 2, (
            f"&SYSTEM nspin should be 2 (spin-polarised for the O2 triplet) for the molecule "
            f"phase: {MOLECULE_PHASE}"
        )
        assert mol.system["tot_magnetization"] == 2, (
            f"&SYSTEM tot_magnetization should be 2 for the O2 triplet ground state "
            f"(molecule phase: {MOLECULE_PHASE})"
        )
        assert mol.system["ntyp"] == 1, (  # single (O) species
            f"&SYSTEM ntyp should be 1 (single O species) for the O2 molecule phase: {MOLECULE_PHASE}"
        )
        assert mol.system["nat"] == 2, (
            f"&SYSTEM nat should be 2 (two O atoms) for the O2 molecule phase: {MOLECULE_PHASE}"
        )



class CompetingPhasesAnalyzerQETestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.cpa = CompetingPhasesAnalyzerQE("MgO", entries=MGO_QE_CP_DIR)
        cls.saved_chempots = loadfn(MGO_SAVED_CHEMPOTS)

    def test_parsing_of_qe_outputs(self):
        """All five competing-phase ``espresso.xml`` outputs are parsed."""
        assert len(self.cpa.entries) == 5
        assert len(self.cpa.espresso_paths) == 5
        assert all(p.endswith(".xml") for p in self.cpa.espresso_paths)
        assert {e.name for e in self.cpa.entries} == {"MgO", "Mg", "O2"}
        assert self.cpa.composition == Composition("MgO")
        # the parsed intrinsic phase diagram is built and contains MgO:
        assert isinstance(self.cpa.intrinsic_phase_diagram, PhaseDiagram)
        assert any(
            e.composition.reduced_formula == "MgO" for e in self.cpa.intrinsic_phase_diagram.entries
        )

    def test_chempots_reproduce_saved_notebook_values(self):
        """
        ``cpa.chempots`` matches the values computed in the notebook and saved
        to ``MgO_chempots_QE.json`` (the full ``limits`` / ``elemental_refs``
        / ``limits_wrt_el_refs`` dict).
        """
        parsed = _canonicalise_chempot_dict(self.cpa.chempots)
        saved = _canonicalise_chempot_dict(self.saved_chempots)
        assert set(parsed) == {"limits", "elemental_refs", "limits_wrt_el_refs"}
        _compare_chempot_dicts(parsed["elemental_refs"], saved["elemental_refs"])
        _compare_chempot_dicts(parsed["limits"], saved["limits"])
        _compare_chempot_dicts(parsed["limits_wrt_el_refs"], saved["limits_wrt_el_refs"])

    def test_extracted_chempot_limit_values(self):
        """
        The two MgO chemical-potential limits (relative to the elemental
        references), as extracted from the notebook:
        """
        elemental_refs = self.cpa.chempots["elemental_refs"]
        assert np.isclose(elemental_refs["Mg"], MGO_ELEMENTAL_REFS["Mg"], atol=1e-4)
        assert np.isclose(elemental_refs["O"], MGO_ELEMENTAL_REFS["O"], atol=1e-4)

        wrt_refs = self.cpa.chempots["limits_wrt_el_refs"]
        # Mg-rich limit: Mg at its reference, O drops by the formation energy:
        assert np.isclose(wrt_refs["MgO-Mg"]["Mg"], 0.0, atol=1e-4)
        assert np.isclose(wrt_refs["MgO-Mg"]["O"], MGO_FORMATION_ENERGY, atol=1e-4)
        # O-rich limit: O at its reference, Mg drops by the formation energy:
        assert np.isclose(wrt_refs["MgO-O2"]["O"], 0.0, atol=1e-4)
        assert np.isclose(wrt_refs["MgO-O2"]["Mg"], MGO_FORMATION_ENERGY, atol=1e-4)

    def test_chempots_df_matches_limits(self):
        """``cpa.chempots_df`` is the per-element Δμ table for both limits."""
        df = self.cpa.chempots_df
        assert set(df.columns) == {"Mg", "O"}
        assert {"MgO-Mg", "MgO-O2"} <= set(df.index)
        assert np.isclose(df.loc["MgO-Mg", "O"], MGO_FORMATION_ENERGY, atol=1e-4)
        assert np.isclose(df.loc["MgO-O2", "Mg"], MGO_FORMATION_ENERGY, atol=1e-4)
        assert np.isclose(df.loc["MgO-Mg", "Mg"], 0.0, atol=1e-4)
        assert np.isclose(df.loc["MgO-O2", "O"], 0.0, atol=1e-4)

    def test_mgo_formation_energy(self):
        """
        The MgO formation energy from the parsed intrinsic phase diagram
        equals the extracted chemical-potential limit.
        """
        pd = self.cpa.intrinsic_phase_diagram
        mgo_entry = next(e for e in pd.entries if e.composition.reduced_formula == "MgO")
        # normalise to per-formula-unit (MgO entry may be a supercell/multiple f.u.):
        form_energy_per_fu = pd.get_form_energy(mgo_entry) / mgo_entry.composition.get_reduced_composition_and_factor()[1]
        assert np.isclose(form_energy_per_fu, MGO_FORMATION_ENERGY, atol=1e-4)

    def test_espresso_none_parsed(self):
        """
        An empty directory contains no QE ``.xml`` outputs to parse, 
        so a ``FileNotFoundError`` is raised.
        """
        with (
            tempfile.TemporaryDirectory() as empty_dir,
            pytest.raises(FileNotFoundError, match=r"No `\.xml` files have been parsed"),
        ):
            CompetingPhasesAnalyzerQE("MgO", entries=empty_dir)

    def test_bulk_not_found(self):
        """
        Requesting a composition whose bulk phase is absent from the supplied QE data (the MgO
        competing phases) raises ``ValueError``.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # QE parsing warnings, irrelevant here
            with pytest.raises(ValueError) as exc:
                CompetingPhasesAnalyzerQE("CrO2", entries=MGO_QE_CP_DIR)
        assert (
            "Could not find bulk phase for CrO2 in the supplied data. Found intrinsic phase diagram "
            "entries for: {'O2'}"
        ) in str(exc.value)

    def test_from_entries_raises_when_host_element_lacks_elemental_reference(self):
        """
        Removing the O2 elemental reference from the parsed MgO entries leaves
        host element O without a reference phase, so rebuilding the analyzer
        from those entries raises ``ValueError``.
        """
        entries_no_o2 = [e for e in self.cpa.entries if e.composition.reduced_formula != "O2"]
        with pytest.raises(
            ValueError,
            match="No elemental reference phase was parsed for host element",
        ):
            CompetingPhasesAnalyzerQE("MgO", entries_no_o2)



class QECompatibilityChecksTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.pristine_entries = CompetingPhasesAnalyzerQE("MgO", entries=MGO_QE_CP_DIR).entries
        # bulk MgO is used as the reference entry for both compatibility checks:
        bulk_entry = next(e for e in cls.pristine_entries if e.name == "MgO")
        cls.ref_basis = bulk_entry.data["qe_input"]["basis"]
        # reference ecutwfc/ecutrho values

    @staticmethod
    def _init_cpa_with_warnings(entries, **kwargs) -> tuple[CompetingPhasesAnalyzerQE, list[str]]:
        """
        Initialise a ``CompetingPhasesAnalyzerQE`` from ``entries``, returning
        it along with any captured "mismatching ..." warning messages.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cpa = CompetingPhasesAnalyzerQE("MgO", entries=entries, **kwargs)
        return cpa, [str(warning.message) for warning in w if "mismatching" in str(warning.message)]

    def _entries_with_mismatches(self) -> tuple[list, ComputedEntry, ComputedEntry, str]:
        """
        Pristine entries with cutoff mismatches introduced on a Mg entry
        (``ecutwfc`` & ``ecutrho``) and on the O2 entry (``ecutrho`` and its
        pseudopotential filename); returns
        ``(entries, mg_entry, o2_entry, original_O_pseudo)``.
        """
        entries = deepcopy(self.pristine_entries)
        for entry in entries:  # clear flags set by the setUpClass parse
            entry.data.pop("mismatching_QE_input_params", None)
            entry.data.pop("mismatching_pseudo_filenames", None)
        mg_entry = next(e for e in entries if e.name == "Mg")
        mg_entry.data["qe_input"]["basis"]["ecutwfc"] = self.ref_basis["ecutwfc"] + 40
        mg_entry.data["qe_input"]["basis"]["ecutrho"] = self.ref_basis["ecutrho"] + 200
        o2_entry = next(e for e in entries if e.name == "O2")
        o2_entry.data["qe_input"]["basis"]["ecutrho"] = self.ref_basis["ecutrho"] + 200
        orig_o_pseudo = o2_entry.data["pseudo_filenames"][0]
        o2_entry.data["pseudo_filenames"] = ["O_custom.UPF"]
        return entries, mg_entry, o2_entry, orig_o_pseudo

    def test_consistent_entries_pass_compatibility_checks(self):
        """The (consistent) example MgO entries produce no compatibility warnings."""
        entries = deepcopy(self.pristine_entries)
        _cpa, mismatch_warnings = self._init_cpa_with_warnings(entries)
        assert not mismatch_warnings
        for entry in entries:  # checks ran and passed on every entry:
            assert entry.data["mismatching_QE_input_params"] is False, entry.name
            assert entry.data["mismatching_pseudo_filenames"] is False, entry.name

    def test_mismatching_pseudos_and_qe_input_params(self):
        """
        Cutoff (``ecutwfc``/``ecutrho``) and pseudopotential mismatches vs the
        bulk reference each give one aggregated warning (covering all
        mismatching entries), and are flagged on the offending entries' data.
        """
        entries, mg_entry, o2_entry, orig_o_pseudo = self._entries_with_mismatches()
        _cpa, mismatch_warnings = self._init_cpa_with_warnings(entries)
        assert len(mismatch_warnings) == 2  # one aggregated warning per check

        ref_ecutwfc, ref_ecutrho = self.ref_basis["ecutwfc"], self.ref_basis["ecutrho"]
        params_message = next(m for m in mismatch_warnings if "QE input parameters" in m)
        assert "Entries ['Mg']" in params_message
        assert "Entries ['O2']" in params_message
        assert f"('ecutwfc', {ref_ecutwfc + 40!r}, {ref_ecutwfc!r})" in params_message
        assert f"('ecutrho', {ref_ecutrho + 200!r}, {ref_ecutrho!r})" in params_message
        assert "Where MgO was used as the reference entry calculation" in params_message

        pseudo_message = next(m for m in mismatch_warnings if "pseudopotential filenames" in m)
        assert f"O2: [('O_custom.UPF', '{orig_o_pseudo}')]" in pseudo_message
        assert "Where MgO was used as the reference entry calculation" in pseudo_message

        # mismatches recorded on the offending entries' data (and only those):
        assert set(mg_entry.data["mismatching_QE_input_params"]) == {
            ("ecutwfc", ref_ecutwfc + 40, ref_ecutwfc),
            ("ecutrho", ref_ecutrho + 200, ref_ecutrho),
        }
        assert o2_entry.data["mismatching_QE_input_params"] == [
            ("ecutrho", ref_ecutrho + 200, ref_ecutrho)
        ]
        assert o2_entry.data["mismatching_pseudo_filenames"] == [("O_custom.UPF", orig_o_pseudo)]
        for entry in entries:
            if entry not in (mg_entry, o2_entry):
                assert entry.data["mismatching_QE_input_params"] is False, entry.name
            if entry is not o2_entry:
                assert entry.data["mismatching_pseudo_filenames"] is False, entry.name

    def test_check_compatibility_false_skips_checks(self):
        """``check_compatibility=False`` skips both checks (no warnings, no flags)."""
        entries, *_mismatched = self._entries_with_mismatches()
        cpa, mismatch_warnings = self._init_cpa_with_warnings(entries, check_compatibility=False)
        assert not mismatch_warnings
        for entry in entries:
            assert "mismatching_QE_input_params" not in entry.data
            assert "mismatching_pseudo_filenames" not in entry.data
        assert cpa.chempots  # chempots still parsed as usual



# each skipif is reported separately, so the skip reason states exactly which
# requirement was not met:
@pytest.mark.skipif(
    not _run_heavy_tests(),
    reason="Skipping heavy (MP network) test: heavy tests disabled "
    "(POTCARs unavailable or DOPED_SKIP_HEAVY_TESTS=true)",
)
@pytest.mark.skipif(not api_key, reason="Skipping: no Materials Project API key available")
class CompetingPhasesQEInputGenerationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orig_cwd = os.getcwd()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.cp = CompetingPhasesQE("MgO", energy_above_hull=0.01, api_key=api_key)
        # generation methods write to the host-named parent folder under the CWD:
        cls.tmp = tempfile.mkdtemp(prefix="doped_cpqe_gen_")
        os.chdir(cls.tmp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.cp.qe_convergence_setup()
            cls.cp.qe_std_setup()
        cls.cp_dir = os.path.join(cls.tmp, DEFAULT_QE_CP_RELPATH)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.orig_cwd)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _generate_std_in_fresh_dir(self, **kwargs) -> str:
        """Run ``qe_std_setup(**kwargs)`` in a throwaway CWD; return its path."""
        sub = tempfile.mkdtemp(prefix="doped_cpqe_var_")
        self.addCleanup(shutil.rmtree, sub, ignore_errors=True)
        self.addCleanup(os.chdir, self.tmp)
        os.chdir(sub)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cp.qe_std_setup(**kwargs)
        return os.path.join(sub, DEFAULT_QE_CP_RELPATH)

    def test_generated_folder_structure(self):
        """Five phase folders, each with std + ecut sweeps; molecule has no k-sweep."""
        phases = {d for d in os.listdir(self.cp_dir) if os.path.isdir(os.path.join(self.cp_dir, d))}
        assert phases == EXPECTED_PHASES

        for phase in phases:
            assert os.path.isfile(os.path.join(self.cp_dir, phase, "espresso_std", "pw.in"))
            ecut_dir = os.path.join(self.cp_dir, phase, "ecut_convergence")
            assert {f"ecutwfc_{e}" for e in EXPECTED_ECUTS} == set(os.listdir(ecut_dir))
            kpoint_dir = os.path.join(self.cp_dir, phase, "kpoint_converge")
            if phase == MOLECULE_PHASE:
                assert not os.path.isdir(kpoint_dir)
            else:
                assert os.path.isdir(kpoint_dir) and os.listdir(kpoint_dir)

    def test_generated_std_inputs_use_sssp_defaults(self):
        """
        Std relax inputs use SSSP defaults (ecutwfc=60, ecutrho=240), with
        Gaussian smearing only on the metallic phases.
        """
        for phase in NON_MOLECULE_PHASES:
            pw = PWin.from_file(os.path.join(self.cp_dir, phase, "espresso_std", "pw.in"))
            _assert_valid_pw_input(pw)
            _assert_sssp_pseudos(pw)
            _assert_smearing(pw, phase)
            assert pw.control["calculation"] == "vc-relax"
            assert pw.system["ecutwfc"] == 60  # SSSP set default
            assert pw.system["ecutrho"] == 240

        # metallic (3 Mg polymorphs) / non-metallic (MgO) / molecular (O2) classification:
        assert len(self.cp.entries) == len(EXPECTED_PHASES)
        assert {e.name for e in self.cp.metallic_entries} == {"Mg"}
        assert len(self.cp.metallic_entries) == sum(_phase_is_metal(p) for p in EXPECTED_PHASES) == 3
        assert {e.name for e in self.cp.nonmetallic_entries} == {"MgO"}
        assert {e.name for e in self.cp.molecular_entries} == {"O2"}
        assert all(e.data.get("molecule") for e in self.cp.molecular_entries)
        assert not any(  # molecule flag only on the molecular entries:
            e.data.get("molecule") for e in self.cp.metallic_entries + self.cp.nonmetallic_entries
        )

    def test_generated_convergence_inputs(self):
        """Generated ecut/k-point sweeps are SCF calcs with the expected grids/cutoffs."""
        mgo = "MgO_Fm-3m_EaH_0"
        ecut_dir = os.path.join(self.cp_dir, mgo, "ecut_convergence")
        for ecut in EXPECTED_ECUTS:
            pw = PWin.from_file(os.path.join(ecut_dir, f"ecutwfc_{ecut}", "pw.in"))
            assert pw.control["calculation"] == "scf"
            assert pw.system["ecutwfc"] == ecut
            assert pw.system["ecutrho"] == 240

        kpoint_dir = os.path.join(self.cp_dir, mgo, "kpoint_converge")
        for kname in os.listdir(kpoint_dir):
            pw = PWin.from_file(os.path.join(kpoint_dir, kname, "pw.in"))
            assert pw.control["calculation"] == "scf"
            assert list(pw.k_points.grid) == _kpoint_grid_from_folder_name(kname)

    def test_generated_molecule_handling(self):
        """O2 std input is a Γ-only spin-polarised fixed-cell relax."""
        pw = PWin.from_file(os.path.join(self.cp_dir, MOLECULE_PHASE, "espresso_std", "pw.in"))
        assert pw.control["calculation"] == "relax"
        assert str(pw.k_points.option) == "gamma"
        assert pw.system["nspin"] == 2
        assert pw.system["tot_magnetization"] == 2

    def test_use_hse_std_setup(self):
        """``qe_std_setup(use_hse=True)`` writes the HSE06 ``&SYSTEM`` defaults."""
        cp_dir = self._generate_std_in_fresh_dir(use_hse=True)
        pw = PWin.from_file(os.path.join(cp_dir, "MgO_Fm-3m_EaH_0", "espresso_std", "pw.in"))
        assert pw.system["input_dft"] == "HSE"
        assert pw.system["exx_fraction"] == 0.25
        assert pw.system["screening_parameter"] == 0.106
        assert pw.system["nqx1"] == pw.system["nqx2"] == pw.system["nqx3"] == 1

    def test_hse_std_setup_user_overrides(self):
        """
        With ``use_hse=True``, ``user_system_settings`` override the HSE06
        ``&SYSTEM`` defaults (``exx_fraction``, ``screening_parameter``, the
        ``nqx`` q-point grid, ...), while HSE keys that are not overridden
        (e.g. ``input_dft = 'HSE'``, ``exxdiv_treatment``) are retained.
        """
        user_system_settings = {
            "exx_fraction": 0.4,  
            "screening_parameter": 0.2,  
            "nqx1": 2,
            "nqx2": 2,
            "nqx3": 2,  
            "exxdiv_treatment" : "vcut_spherical",
        }
        cp_dir = self._generate_std_in_fresh_dir(
            use_hse=True, user_system_settings=user_system_settings
        )
        pw = PWin.from_file(os.path.join(cp_dir, "MgO_Fm-3m_EaH_0", "espresso_std", "pw.in"))
        _assert_valid_pw_input(pw)

        for key, val in user_system_settings.items():
            assert pw.system[key] == val, f"&SYSTEM {key}"

    def test_ions_and_cell_overrides(self):
        """
        ``user_ions_settings`` / ``user_cell_settings`` override the ``&IONS`` /
        ``&CELL`` namelists in the std relaxation inputs. For solid phases
        (``vc-relax``) both namelists are written and receive the overrides;
        for molecules (fixed-cell ``relax``) ``&CELL`` is not written, so its
        overrides are dropped while ``&IONS`` still applies.
        """
        user_ions_settings = {"ion_dynamics": "damp", "pot_extrapolation": "none"}
        user_cell_settings = {"cell_dynamics": "damp-pr", "cell_dofree": "a", "press": 10}

        cp_dir = self._generate_std_in_fresh_dir(
            user_ions_settings=user_ions_settings,
            user_cell_settings=user_cell_settings,
        )

       
        pw = PWin.from_file(os.path.join(cp_dir, "MgO_Fm-3m_EaH_0", "espresso_std", "pw.in"))
        _assert_valid_pw_input(pw)
        assert pw.control["calculation"] == "vc-relax"
        for key, val in user_ions_settings.items():
            assert pw.ions[key] == val, f"&IONS {key}"
        for key, val in user_cell_settings.items():
            assert pw.cell[key] == val, f"&CELL {key}"

    
        mol = PWin.from_file(os.path.join(cp_dir, MOLECULE_PHASE, "espresso_std", "pw.in"))
        _assert_valid_pw_input(mol)
        assert mol.control["calculation"] == "relax"
        for key, val in user_ions_settings.items():
            assert mol.ions[key] == val, f"&IONS {key}"
        assert mol.cell is None  # &CELL is not written for fixed-cell (molecule) relaxations

    def test_pseudo_map_and_dir_overrides(self):
        """``pseudo_map`` / ``pseudo_dir`` override the SSSP defaults."""
        cp_dir = self._generate_std_in_fresh_dir(
            pseudo_map={"O": "O_custom.UPF"}, pseudo_dir="./my_pseudos/"
        )
        pw = PWin.from_file(os.path.join(cp_dir, "MgO_Fm-3m_EaH_0", "espresso_std", "pw.in"))
        files = dict(zip(pw.atomic_species.symbols, pw.atomic_species.files))
        assert files["O"] == "O_custom.UPF"
        assert files["Mg"] == qe_SSSP_pseudo_filenames["Mg"]  # untouched element keeps SSSP default
        assert pw.control["pseudo_dir"] == "./my_pseudos/"

    def test_user_namelist_overrides(self):
        """
        ``user_{system,control,electron}_settings`` override each of the three
        user-overridable QE namelists (``&SYSTEM``, ``&CONTROL``,
        ``&ELECTRONS``).
        """
        # keys chosen to not collide with anything generation sets after the merge:
        user_system_settings = {"nbnd": 42, "tot_charge": 1.0}
        user_control_settings = {"verbosity": "high", "nstep": 123}
        user_electron_settings = {"conv_thr": 1e-9, "mixing_beta": 0.6}

        cp_dir = self._generate_std_in_fresh_dir(
            user_system_settings=user_system_settings,
            user_control_settings=user_control_settings,
            user_electron_settings=user_electron_settings,
        )
        pw = PWin.from_file(os.path.join(cp_dir, "MgO_Fm-3m_EaH_0", "espresso_std", "pw.in"))
        _assert_valid_pw_input(pw)  

        # each override landed in its own namelist:
        for key, val in user_system_settings.items():
            assert pw.system[key] == val, f"&SYSTEM {key}"
        for key, val in user_control_settings.items():
            assert pw.control[key] == val, f"&CONTROL {key}"
        for key, val in user_electron_settings.items():
            assert pw.electrons[key] == val, f"&ELECTRONS {key}"

        # generation-set keys survive the user overrides:
        assert pw.system["ibrav"] == 0
        assert pw.system["nat"] == 2 and pw.system["ntyp"] == 2
        assert pw.control["calculation"] == "vc-relax"




# QE bulk DOS (dos.x tetrahedra output) + bulk supercell xml for the FermiDos:
MGO_QE_FILDOS = os.path.join(EXAMPLE_DIR, "MgO_qe", "DOS", "MgO_bulk_coarse_tetrahedra.dos")
MGO_QE_BULK_XML = os.path.join(EXAMPLE_DIR, "MgO_qe", "Defects", "MgO_bulk", "espresso_std", "espresso.xml")

MGO_QE_DEFECT_DICT = os.path.join(EXAMPLE_DIR, "MgO_qe", "Defects", "MgO_defect_dict.json")

MGO_QE_BAND_GAP = 4.8099  # bulk supercell eigenvalue gap, supplied explicitly in the notebook
# expected properties of the QE (FermiDos) bulk DOS:
MGO_QE_DOS_NELECS = 864  # QE ``nelec`` of the 216-atom bulk supercell
MGO_QE_DOS_VBM, MGO_QE_DOS_CBM = 5.7880, 10.4907  # eV; DOS-grid band edges (0.1 eV dos.x grid)


def if_present_rm(path):
    if os.path.exists(path):
        os.remove(path)


class QEDefectThermodynamicsTestCase(unittest.TestCase):
    """
    Testing ``DefectThermodynamics`` analyses with Quantum ESPRESSO data
    The thermodynamics object is built from the pre-parsed beta=0.5 (optimal)
    defect dict, the saved QE chemical potentials (validated against
    ``CompetingPhasesAnalyzerQE`` in the tests above) and a ``FermiDos`` from
    the QE bulk DOS.
    """

    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # DOS-grid band-edge mismatch warning tested separately
            cls.bulk_dos = get_fermi_dos_from_espresso_dos(MGO_QE_FILDOS, bulk_pwxml=MGO_QE_BULK_XML)
            cls.defect_dict = loadfn(MGO_QE_DEFECT_DICT)
            cls.chempots = loadfn(MGO_SAVED_CHEMPOTS)
            cls.thermo = DefectThermodynamics(
                cls.defect_dict,
                chempots=cls.chempots,
                band_gap=MGO_QE_BAND_GAP,
                bulk_dos=cls.bulk_dos,
            )

    def tearDown(self):
        import matplotlib.pyplot as plt

        plt.close("all")

    # ── FermiDos construction from the QE DOS ────────────────────────────────

    def test_fermi_dos_from_espresso_dos(self):
        """
        ``get_fermi_dos_from_espresso_dos`` builds a ``FermiDos`` with the bulk
        supercell structure (for cm^-3 normalisation), QE ``nelec``
        normalisation and the DOS-determined band edges.
        """
        dos = self.bulk_dos
        assert isinstance(dos, FermiDos)
        assert dos.nelecs == MGO_QE_DOS_NELECS
        assert dos.structure.composition.reduced_formula == "MgO"
        assert len(dos.structure) == 216  # bulk supercell from espresso.xml
        cbm, vbm = dos.get_cbm_vbm()
        assert np.isclose(vbm, MGO_QE_DOS_VBM, atol=1e-4)
        assert np.isclose(cbm, MGO_QE_DOS_CBM, atol=1e-4)

    def test_fermi_dos_from_bulk_pwxml_object(self):
        """
        Test written for fermi dos object creation using bulk dos and bulk pwxml file.
        """
        from pymatgen.io.espresso.outputs.pwxml import PWxml

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pwxml = PWxml(MGO_QE_BULK_XML)
            dos_from_object = get_fermi_dos_from_espresso_dos(MGO_QE_FILDOS, bulk_pwxml=pwxml)
            dos_explicit = get_fermi_dos_from_espresso_dos(
                MGO_QE_FILDOS, structure=pwxml.final_structure, nelecs=pwxml.nelec
            )

        # xml-derived normalisation inputs:
        assert pwxml.nelec == MGO_QE_DOS_NELECS  # QE valence electron count
        assert len(pwxml.final_structure) == 216  # bulk supercell

        for dos in (dos_from_object, dos_explicit):
            assert isinstance(dos, FermiDos)
            assert dos.nelecs == self.bulk_dos.nelecs == MGO_QE_DOS_NELECS
            assert dos.structure == self.bulk_dos.structure
            assert np.isclose(dos.volume, pwxml.final_structure.volume)  # cm^-3 normalisation volume
            assert np.allclose(dos.energies, self.bulk_dos.energies)
            assert np.allclose(dos.get_densities(), self.bulk_dos.get_densities())

    def test_fermi_dos_requires_structure(self):
        """Without ``bulk_pwxml``/``structure`` there is no volume -> error."""
        with pytest.raises(ValueError, match="bulk structure is required"):
            get_fermi_dos_from_espresso_dos(MGO_QE_FILDOS)

    def test_fermi_dos_warns_without_nelecs(self):
        """
        With a structure but no ``bulk_pwxml``/``nelecs``, the DOS is
        normalised to the all-electron count, with a warning.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dos = get_fermi_dos_from_espresso_dos(MGO_QE_FILDOS, structure=self.bulk_dos.structure)
        assert any("number of valence electrons" in str(warn.message) for warn in w)
        assert dos.nelecs != MGO_QE_DOS_NELECS  # all-electron count instead of nelec


    def test_defect_thermodynamics_setup(self):
        """Basic attributes of the QE ``DefectThermodynamics`` object."""
        assert set(self.thermo.defect_entries) == {
            "Mg_O_+1",
            "Mg_O_+2",
            "Mg_O_+3",
            "Mg_O_+4",
        }
        assert self.thermo.band_gap == MGO_QE_BAND_GAP
        assert np.isclose(self.thermo.vbm, 5.7145, atol=1e-4)  # from the QE defect entries
        assert isinstance(self.thermo.bulk_dos, FermiDos)
        assert {"MgO-Mg", "MgO-O2"} == set(self.thermo.chempots["limits"])

    def test_transition_levels(self):
        """
        The Mg_O in-gap (thermodynamic) transition levels underlying the defect
        level diagram: ε(+4/+3) ≈ 0.22 eV, ε(+3/+2) ≈ 0.90 eV and
        ε(+2/+1) ≈ 4.28 eV above the VBM.
        """
        tl_map = self.thermo.transition_level_map
        assert set(tl_map) == {"Mg_O"}  # charge states grouped into one defect
        tls = {tuple(charges): energy for energy, charges in tl_map["Mg_O"].items()}
        assert set(tls) == {(4, 3), (3, 2), (2, 1)}
        assert np.isclose(tls[(4, 3)], 0.2178, atol=1e-4)
        assert np.isclose(tls[(3, 2)], 0.9030, atol=1e-4)
        assert np.isclose(tls[(2, 1)], 4.2836, atol=1e-4)
        for energy in tls.values():  # all in-gap:
            assert 0 < energy < MGO_QE_BAND_GAP

    def test_defect_level_diagram_plot(self):
        """
        ``thermo.plot(limit="MgO-Mg", chempot_table=False)`` produces the
        formation-energy / defect-level diagram spanning the band gap.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = self.thermo.plot(limit="MgO-Mg", chempot_table=False)
        ax = fig.gca()
        assert ax.get_xlabel() == "Fermi Level (eV)"
        assert ax.get_ylabel() == "Formation Energy (eV)"
        # x-range spans the band gap (with the doped default 0.3 eV padding):
        assert np.allclose(ax.get_xlim(), (-0.3, MGO_QE_BAND_GAP + 0.3), atol=0.01)
        assert len(ax.get_lines()) >= 1  # Mg_O formation energy line(s) drawn
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert legend_labels == ["Mg$_{O}$"]

    def test_symmetries_and_degeneracies(self):
        """
        Point symmetries and spin/orientational degeneracies of the relaxed
        Mg_O defects (notebook: ``get_symmetries_and_degeneracies()``).
        """
        df = self.thermo.get_symmetries_and_degeneracies()
        assert set(df["Site_Symm"]) == {"Oh"}  # all on the octahedral O site
        expected = {  # q: (Defect_Symm, g_Orient, g_Spin)
            ("Mg_O", "+4"): ("C2v", 12, 1),
            ("Mg_O", "+3"): ("C3v", 8, 2),
            ("Mg_O", "+2"): ("C3v", 8, 1),
            ("Mg_O", "+1"): ("Cs", 24, 2),
            ("Mg_O_Unrelaxed", "+1"): ("Oh", 1, 2),
        }
        for (defect, q), (symm, g_orient, g_spin) in expected.items():
            row = df.loc[(defect, q)]
            assert row["Defect_Symm"] == symm, f"{defect} {q}"
            assert row["g_Orient"] == g_orient, f"{defect} {q}"
            assert row["g_Spin"] == g_spin, f"{defect} {q}"
            assert row["g_Total"] == g_orient * g_spin, f"{defect} {q}"


    def test_annealed_carrier_concentrations(self):
        """
        Self-consistent carrier concentrations with the QE data (beta=0.5
        corrections, coarse-grid bulk DOS) for Mg-rich MgO, annealed at 1400 K
        and quenched to 300 K: n = 5.57e-5 cm^-3 and p = 2.87e-39 cm^-3
        (E_F = 3.50 eV wrt VBM).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fermi_level, e_conc, h_conc, _conc_df = self.thermo.get_fermi_level_and_concentrations(
                limit="Mg-rich", annealing_temperature=1400
            )
        assert np.isclose(e_conc, 5.57e-05, rtol=0.01), f"n = {e_conc:.3e} cm^-3"
        assert np.isclose(np.log10(h_conc), np.log10(2.874e-39), atol=1e-4), f"p = {h_conc:.3e} cm^-3"
        assert np.isclose(fermi_level, 3.4958, atol=1e-4), f"E_F = {fermi_level:.3f} eV"


if __name__ == "__main__":
    unittest.main()

