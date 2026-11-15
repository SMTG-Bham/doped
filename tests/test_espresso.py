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
import unittest
import warnings

import numpy as np

from doped.io.espresso.outputs import BOHR_TO_ANGSTROM, PAW_COMPENSATION_RADII
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
