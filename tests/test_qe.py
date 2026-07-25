"""
Tests for the ``doped.qe`` module (Quantum ESPRESSO ``pw.x`` input-file writing).

"""

import os
import shutil
import tempfile
import unittest
import warnings

from pymatgen.core import Element, Structure
from pymatgen.io.espresso.inputs.pwin import PWin
from pymatgen.io.espresso.outputs import PWxml
from test_utils import EXAMPLE_DIR

from doped.generation import DefectsGenerator
from doped.qe import (
    _build_qe_base_settings,
    _kpoints_grid_from_reciprocal_density,
    _write_qe_pw_input,
    default_qe_HSE_set,
    default_qe_SSSP_set,
    qe_SSSP_pseudo_filenames,
    qe_convergence_setup_from_structure,
    qe_defect_setup_from_generator,
    qe_relax_setup_from_structure,
)
from doped.utils.parsing import _get_defect_supercell

MGO_QE_DIR = os.path.join(EXAMPLE_DIR, "MgO_qe")


def _mgo_primitive() -> Structure:
    """
    Relaxed MgO rocksalt primitive cell from the ``MgO_qe`` example (the
    ``MgO_Fm-3m`` competing-phase ``vc-relax`` output).
    """
    xml = os.path.join(
        MGO_QE_DIR, "Competing_phases_worked_examples", "MgO_Fm-3m_EaH_0", "espresso_std", "espresso.xml"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PWxml(xml).final_structure


class QEInputWritingTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mgo_prim = _mgo_primitive()
        cls.tmp = tempfile.mkdtemp(prefix="doped_qe_test_")
        cls.ecutwfc = 70
        cls.kpoint_density = 60

        # generate the defect supercells once (the slow step) and reuse:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.defect_gen = DefectsGenerator(
                structure=cls.mgo_prim, supercell_gen_kwargs={"force_cubic": True}
            )


        cls.conv_dir = os.path.join(cls.tmp, "MgO_convergence")
        cls.conv_written = qe_convergence_setup_from_structure(
            cls.mgo_prim,
            output_dir=cls.conv_dir,
            kpoint_density_range=(20, 80, 20),
            ecut_range=(40, 60, 10),
            ecut_sweep_kpoint_density=40,
        )


        cls.relax_dir = os.path.join(cls.tmp, "MgO_bulk_relax")
        cls.relax_path = qe_relax_setup_from_structure(
            cls.mgo_prim,
            ecutwfc=cls.ecutwfc,
            kpoint_density=cls.kpoint_density,
            output_dir=cls.relax_dir,
        )


        cls.defects_dir = os.path.join(cls.tmp, "MgO_defects")
        cls.defect_written = qe_defect_setup_from_generator(
            cls.defect_gen,
            ecutwfc=cls.ecutwfc,
            kpoint_density=cls.kpoint_density,
            output_dir=cls.defects_dir,
            include_bulk=True,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_convergence_folder_structure(self):
        """
        Convergence sweeps write flat ``<param_folder>/pw.in`` (no espresso_std),
        with the ``ecutwfc`` sweep varying ``ecutwfc`` at a fixed k-grid, and the
        k-point sweep varying the k-grid at a fixed ``ecutwfc`` (both ``scf``).
        """
        assert set(self.conv_written) == {"kpoint_converge", "ecut_convergence"}
        assert len(self.conv_written["ecut_convergence"]) == 3
        assert len(self.conv_written["kpoint_converge"]) == 3

        for path in self.conv_written["kpoint_converge"] + self.conv_written["ecut_convergence"]:
            assert os.path.isfile(path), f"missing {path}"
            assert os.path.basename(path) == "pw.in", f"input file not named pw.in: {path}"


        # the two sub-trees exist with the documented parent folder names
        assert os.path.isdir(os.path.join(self.conv_dir, "kpoint_converge"))
        assert os.path.isdir(os.path.join(self.conv_dir, "ecut_convergence"))

        fixed_ecut_kgrid = _kpoints_grid_from_reciprocal_density(self.mgo_prim, 40)
        ecut_grids = []
        for ecut in (40, 50, 60):
            pw = PWin.from_file(
                os.path.join(self.conv_dir, "ecut_convergence", f"ecutwfc_{ecut}", "pw.in")
            )
            assert pw.control["calculation"] == "scf", f"calculation should be scf (ecutwfc_{ecut})"
            # ecutwfc matches the folder name:
            assert pw.system["ecutwfc"] == ecut, f"ecutwfc mismatch: {pw.system['ecutwfc']} != {ecut}"
            ecut_grids.append([int(k) for k in pw.k_points.grid])
        # k-grid is constant across the whole ecutwfc sweep, at the fixed sweep density:
        assert all(grid == fixed_ecut_kgrid for grid in ecut_grids), ecut_grids
        
        expected_kgrids, seen = [], set()
        for density in range(20, 80, 20):
            grid = tuple(_kpoints_grid_from_reciprocal_density(self.mgo_prim, density))
            if grid not in seen:  # duplicate grids from nearby densities are skipped
                seen.add(grid)
                expected_kgrids.append(grid)

        kpoint_grids = []
        for path in self.conv_written["kpoint_converge"]:
            pw = PWin.from_file(path)
            assert pw.control["calculation"] == "scf", f"calculation should be scf (input: {path})"
            # fixed at the set default across the k-grid sweep:
            assert pw.system["ecutwfc"] == 60, f"ecutwfc should be 60 (input: {path})"
            kpoint_grids.append(tuple(int(k) for k in pw.k_points.grid))

        assert kpoint_grids == expected_kgrids

    def test_relax_folder_structure(self):
        """Standalone relaxation writes ``<output_dir>/espresso_std/pw.in``."""
        assert self.relax_path == os.path.join(self.relax_dir, "espresso_std", "pw.in")
        assert os.path.isfile(self.relax_path), f"missing relax input: {self.relax_path}"

    def test_defect_folder_structure(self):
        """Every defect (and the bulk reference) lives in its own espresso_std subfolder."""
        expected_keys = {*self.defect_gen.defect_entries, "MgO_bulk"}
        assert set(self.defect_written) == expected_keys

        for name, path in self.defect_written.items():
            assert path == os.path.join(self.defects_dir, name, "espresso_std", "pw.in"), (
                f"unexpected output path for {name!r}: {path}"
            )
            assert os.path.isfile(path), f"missing input file for {name!r}: {path}"

        # bulk reference explicitly present
        bulk_pw = os.path.join(self.defects_dir, "MgO_bulk", "espresso_std", "pw.in")
        assert os.path.isfile(bulk_pw), f"missing bulk reference input: {bulk_pw}"

    def test_default_host_named_output_paths(self):
        """
        With no ``output_dir``, all three helpers write under a shared
        ``{host}_QE`` parent (VASP-style ``Bulk_convergence`` / ``Bulk_relax``
        / ``Defects`` subfolders, bulk reference ``{formula}_bulk``), relative
        to the current working directory.
        """
        host = "MgO_QE"
        orig_cwd = os.getcwd()
        sub = tempfile.mkdtemp(prefix="doped_qe_default_")
        try:
            os.chdir(sub)


            conv = qe_convergence_setup_from_structure(
                self.mgo_prim,
                kpoint_density_range=(20, 40, 20),
                ecut_range=(40, 40, 10),
                ecut_sweep_kpoint_density=40,
            )
            conv_root = os.path.join(host, "Bulk_convergence")
            for path in conv["kpoint_converge"] + conv["ecut_convergence"]:
                assert path.startswith(conv_root + os.sep), f"path not under {conv_root}: {path}"
                assert os.path.isfile(os.path.join(sub, path)), f"missing input file: {path}"

            relax = qe_relax_setup_from_structure(self.mgo_prim, ecutwfc=40, kpoint_density=20)
            assert relax == os.path.join(host, "Bulk_relax", "espresso_std", "pw.in")
            assert os.path.isfile(os.path.join(sub, relax)), f"missing relax input: {relax}"

            written = qe_defect_setup_from_generator(
                self.defect_gen, ecutwfc=40, kpoint_density=20, include_bulk=True
            )
            assert "MgO_bulk" in written
            for name, path in written.items():
                assert path == os.path.join(host, "Defects", name, "espresso_std", "pw.in"), (
                    f"unexpected output path for {name!r}: {path}"
                )
                assert os.path.isfile(os.path.join(sub, path)), f"missing input file for {name!r}: {path}"
        finally:
            os.chdir(orig_cwd)
            shutil.rmtree(sub, ignore_errors=True)


    def test_qe_set_defaults_match_yaml(self):
        """The bundled QE sets carry the expected default parameters."""
        # SSSP / GGA convergence set
        assert default_qe_SSSP_set["control"]["calculation"] == "scf"
        assert default_qe_SSSP_set["control"]["tprnfor"] is True
        assert default_qe_SSSP_set["control"]["tstress"] is True
        assert default_qe_SSSP_set["system"]["ecutwfc"] == 60
        assert default_qe_SSSP_set["system"]["ecutrho"] == 240
        assert float(default_qe_SSSP_set["electrons"]["conv_thr"]) == 1e-8
        # empty &IONS / &CELL namelists are intentionally retained
        assert default_qe_SSSP_set["ions"] == {}
        assert default_qe_SSSP_set["cell"] == {}

        # HSE06 hybrid set
        assert default_qe_HSE_set["control"]["calculation"] == "vc-relax"
        assert default_qe_HSE_set["system"]["input_dft"] == "HSE"
        assert default_qe_HSE_set["system"]["exx_fraction"] == 0.25
        assert default_qe_HSE_set["system"]["screening_parameter"] == 0.106
        assert default_qe_HSE_set["system"]["exxdiv_treatment"] == "gygi-baldereschi"
        assert (
            default_qe_HSE_set["system"]["nqx1"]
            == default_qe_HSE_set["system"]["nqx2"]
            == default_qe_HSE_set["system"]["nqx3"]
            == 1
        )

    def test_default_values_written_to_relax_input(self):
        """Defaults from the GGA set are written to a (vc-)relax ``pw.in``."""
        pw = PWin.from_file(self.relax_path)
        # control defaults
        assert pw.control["tprnfor"] is True
        assert pw.control["tstress"] is True
        assert pw.control["calculation"] == "vc-relax"  # standalone relax default
        assert pw.control["pseudo_dir"] == "./pseudo_folder_name/"
        # system: ecutwfc is the user-passed value, ecutrho keeps the set default
        assert pw.system["ecutwfc"] == self.ecutwfc
        assert pw.system["ecutrho"] == 240
        assert pw.system["ibrav"] == 0
        # electrons default
        assert float(pw.electrons["conv_thr"]) == 1e-8

    def test_hse_defaults_written(self):
        """``use_hse=True`` writes the HSE06 ``&SYSTEM`` defaults."""
        hse_path = qe_relax_setup_from_structure(
            self.mgo_prim,
            ecutwfc=self.ecutwfc,
            kpoint_density=self.kpoint_density,
            output_dir=os.path.join(self.tmp, "MgO_hse_relax"),
            use_hse=True,
        )
        sysnl = PWin.from_file(hse_path).system
        assert sysnl["input_dft"] == "HSE"
        assert sysnl["exx_fraction"] == 0.25
        assert sysnl["screening_parameter"] == 0.106
        assert sysnl["nqx1"] == sysnl["nqx2"] == sysnl["nqx3"] == 1

    def test_soc_defaults_written(self):
        """``soc=True`` enables spin orbit coupling settings in the &SYSTEM tag."""
        soc_path = qe_relax_setup_from_structure(
            self.mgo_prim,
            ecutwfc=self.ecutwfc,
            kpoint_density=self.kpoint_density,
            output_dir=os.path.join(self.tmp, "MgO_soc_relax"),
            use_hse=False,
            soc = True,
        )
        sysnl = PWin.from_file(soc_path).system
        assert sysnl["noncolin"] == True
        assert sysnl["lspinorb"] == True


    def test_required_namelists_and_cards_present(self):
        """A defect ``pw.in`` contains all required namelists, cards and key params."""
        any_defect = next(n for n in self.defect_written if n != "MgO_bulk")
        pw = PWin.from_file(self.defect_written[any_defect])

        # namelists present (control/system/electrons always; ions for relax)
        for nl in ("control", "system", "electrons", "ions"):
            assert pw.namelists[nl] is not None, f"&{nl.upper()} missing"

        # required cards present
        for card in ("atomic_species", "atomic_positions", "k_points", "cell_parameters"):
            assert pw.cards[card] is not None, f"{card} card missing"

        # key parameters present in each namelist
        for key in ("calculation", "pseudo_dir"):
            assert key in pw.control
        for key in ("ecutwfc", "ecutrho", "ibrav", "nat", "ntyp"):
            assert key in pw.system
        assert "conv_thr" in pw.electrons


        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert pw.validate() is True

    def test_atomic_cards_use_explicit_lattice(self):
        """``ibrav=0`` with an angstrom CELL_PARAMETERS + ATOMIC_POSITIONS card."""
        pw = PWin.from_file(self.relax_path)
        assert pw.system["ibrav"] == 0
        assert str(pw.cell_parameters.option) == "angstrom"
        assert str(pw.atomic_positions.option) == "angstrom"
        assert pw.system["nat"] == len(self.mgo_prim)
        assert pw.system["ntyp"] == len({s.symbol for s in self.mgo_prim.species})

    def test_sssp_pseudopotentials_and_masses(self):
        """ATOMIC_SPECIES uses the bundled SSSP filenames and correct masses."""
        pw = PWin.from_file(self.relax_path)
        files = dict(zip(pw.atomic_species.symbols, pw.atomic_species.files))
        assert files["Mg"] == qe_SSSP_pseudo_filenames["Mg"]
        assert files["O"] == qe_SSSP_pseudo_filenames["O"]

        masses = dict(zip(pw.atomic_species.symbols, pw.atomic_species.masses))
        for sym, mass in masses.items():
            expected_mass = float(Element(sym).atomic_mass)
            assert abs(mass - expected_mass) < 1e-4, (
                f"atomic mass mismatch for {sym}: {mass} != {expected_mass}"
            )

        # species are sorted by atomic number (O before Mg)
        assert pw.atomic_species.symbols == ["O", "Mg"]

    def test_pseudo_map_override(self):
        """``pseudo_map`` overrides the SSSP default filename for an element."""
        path = qe_relax_setup_from_structure(
            self.mgo_prim,
            ecutwfc=self.ecutwfc,
            kpoint_density=self.kpoint_density,
            output_dir=os.path.join(self.tmp, "MgO_pseudo_override"),
            pseudo_map={"O": "O_custom.UPF"},
        )
        species = PWin.from_file(path).atomic_species
        files = dict(zip(species.symbols, species.files))
        assert files["O"] == "O_custom.UPF"
        assert files["Mg"] == qe_SSSP_pseudo_filenames["Mg"]

    def test_ions_and_cell_overrides(self):
        """
        ``user_ions_settings`` / ``user_cell_settings`` override the ``&IONS`` /
        ``&CELL`` namelists in relaxation inputs. For a variable-cell relaxation
        (``vc-relax``) both namelists are written and receive the overrides; for
        a fixed-cell relaxation (``relax``) ``&CELL`` is not written, so its
        overrides are dropped while ``&IONS`` still applies.
        """
        user_ions_settings = {"ion_dynamics": "damp", "pot_extrapolation": "none"}
        user_cell_settings = {"cell_dynamics": "damp-pr", "cell_dofree": "a", "press": 10}

        # variable-cell relaxation: both &IONS and &CELL written and overridden:
        vc_path = qe_relax_setup_from_structure(
            self.mgo_prim,
            ecutwfc=self.ecutwfc,
            kpoint_density=self.kpoint_density,
            output_dir=os.path.join(self.tmp, "MgO_vc_relax_override"),
            calculation="vc-relax",
            user_ions_settings=user_ions_settings,
            user_cell_settings=user_cell_settings,
        )
        pw = PWin.from_file(vc_path)
        assert pw.control["calculation"] == "vc-relax"
        for key, val in user_ions_settings.items():
            assert pw.ions[key] == val, f"&IONS {key} mismatch: {pw.ions[key]} != {val}"
        for key, val in user_cell_settings.items():
            assert pw.cell[key] == val, f"&CELL {key} mismatch: {pw.cell[key]} != {val}"

        # fixed-cell relaxation: &IONS overridden, &CELL not written:
        fixed_cell_path = qe_relax_setup_from_structure(
            self.mgo_prim,
            ecutwfc=self.ecutwfc,
            kpoint_density=self.kpoint_density,
            output_dir=os.path.join(self.tmp, "MgO_relax_override"),
            calculation="relax",
            user_ions_settings=user_ions_settings,
            user_cell_settings=user_cell_settings,
        )
        fixed_cell_pw = PWin.from_file(fixed_cell_path)
        assert fixed_cell_pw.control["calculation"] == "relax"
        for key, val in user_ions_settings.items():
            assert fixed_cell_pw.ions[key] == val, (
                f"&IONS {key} mismatch: {fixed_cell_pw.ions[key]} != {val}"
            )
        assert fixed_cell_pw.cell is None  # &CELL is not written for fixed-cell relaxations

    def test_defect_ions_override(self):
        """
        ``user_ions_settings`` is applied to every defect (and bulk) supercell
        input in ``qe_defect_setup_from_generator`` (fixed-cell ``relax``, so
        no ``&CELL`` overrides are exposed there).
        """
        written = qe_defect_setup_from_generator(
            self.defect_gen,
            ecutwfc=self.ecutwfc,
            kpoint_density=self.kpoint_density,
            output_dir=os.path.join(self.tmp, "MgO_defects_ions_override"),
            user_ions_settings={"ion_dynamics": "damp", "pot_extrapolation": "none"},
            include_bulk=True,
        )
        for name, path in written.items():
            pw = PWin.from_file(path)
            assert pw.control["calculation"] == "relax", f"calculation mismatch (input: {name})"
            assert pw.ions["ion_dynamics"] == "damp", f"&IONS ion_dynamics mismatch (input: {name})"
            assert pw.ions["pot_extrapolation"] == "none", (
                f"&IONS pot_extrapolation mismatch (input: {name})"
            )
            # &CELL is not written for fixed-cell relaxations:
            assert pw.cell is None, f"&CELL should be dropped (input: {name})"


    def test_bulk_defect_parameter_consistency(self):
        """Bulk and all defect inputs share cutoffs, k-grid and pseudos; only
        ``tot_charge`` and ``nat`` vary."""
        parsed = {name: PWin.from_file(path) for name, path in self.defect_written.items()}

        # reference values taken from the bulk input
        bulk = parsed["MgO_bulk"]
        ref_ecutwfc = bulk.system["ecutwfc"]
        ref_ecutrho = bulk.system["ecutrho"]
        ref_kgrid = list(bulk.k_points.grid)
        ref_pseudos = dict(zip(bulk.atomic_species.symbols, bulk.atomic_species.files))

        assert ref_ecutwfc == self.ecutwfc
        assert ref_ecutrho == 240

        for name, pw in parsed.items():
            msg = f"(input: {name})"
            assert pw.system["ecutwfc"] == ref_ecutwfc, f"ecutwfc mismatch {msg}"
            assert pw.system["ecutrho"] == ref_ecutrho, f"ecutrho mismatch {msg}"
            assert list(pw.k_points.grid) == ref_kgrid, f"k_points grid mismatch {msg}"
            assert str(pw.k_points.option) == "automatic", f"k_points option mismatch {msg}"
            assert dict(zip(pw.atomic_species.symbols, pw.atomic_species.files)) == ref_pseudos, (
                f"atomic_species pseudos mismatch {msg}"
            )
            assert pw.system["ibrav"] == 0, f"ibrav mismatch {msg}"
            assert pw.system["ntyp"] == 2, f"ntyp mismatch {msg}"  # MgO always has both species
            # fixed-cell relaxation for both defects and bulk reference:
            assert pw.control["calculation"] == "relax", f"calculation mismatch {msg}"
            assert pw.cell is None, f"&CELL should be dropped {msg}"  # &CELL dropped for fixed-cell calcs

    def test_defect_tot_charge_and_nat(self):
        """``tot_charge`` matches each entry's charge state; ``nat`` matches its supercell."""
        for name, path in self.defect_written.items():
            pw = PWin.from_file(path)
            if name == "MgO_bulk":
                expected_charge = 0
                expected_nat = len(self.defect_gen.bulk_supercell)
            else:
                entry = self.defect_gen.defect_entries[name]
                expected_charge = entry.charge_state
                expected_nat = len(_get_defect_supercell(entry))

            actual_charge = int(pw.system.get("tot_charge", 0))
            assert actual_charge == expected_charge, (
                f"tot_charge mismatch for {name!r}: {actual_charge} != {expected_charge}"
            )
            assert pw.system["nat"] == expected_nat, (
                f"nat mismatch for {name!r}: {pw.system['nat']} != {expected_nat}"
            )

    def test_defect_starting_magnetization(self):
        """Every defect input seeds ``starting_magnetization=0.1`` per species with
        ``nspin=2`` by default; the bulk reference stays non-spin-polarised;
        ``None`` disables it."""
        # default (starting_magnetization=0.1) was used for self.defect_written:
        for name, path in self.defect_written.items():
            pw = PWin.from_file(path)
            if name == "MgO_bulk":  # bulk reference is excluded from spin polarisation
                assert "starting_magnetization" not in pw.system, (
                    f"bulk reference should not be spin-polarised (input: {name})"
                )
                assert "nspin" not in pw.system, f"bulk reference should not set nspin (input: {name})"
                continue
            ntyp = pw.system["ntyp"]
            assert pw.system["nspin"] == 2, f"nspin should be 2 (input: {name})"
            assert pw.system["starting_magnetization"] == [0.1] * ntyp, (
                f"starting_magnetization mismatch (input: {name})"
            )

        # testing non spin-polarised case:
        no_mag = qe_defect_setup_from_generator(
            self.defect_gen,
            ecutwfc=self.ecutwfc,
            kpoint_density=self.kpoint_density,
            output_dir=os.path.join(self.tmp, "MgO_defects_no_mag"),
            include_bulk=True,
            starting_magnetization=None,
        )
        for name, path in no_mag.items():
            pw = PWin.from_file(path)
            assert "starting_magnetization" not in pw.system, (
                f"starting_magnetization should be absent (input: {name})"
            )
            assert "nspin" not in pw.system, f"nspin should be absent (input: {name})"

    def test_soc_defect_drops_nspin(self):
        """
        ``soc=True`` writes ``noncolin``/``lspinorb`` to every defect (and the
        bulk) input and, crucially, does *not* write ``nspin`` (QE forbids
        ``nspin`` with noncolinear calcs) even though ``starting_magnetization``
        is still seeded on the (spin-polarised) defect supercells. The neutral
        bulk reference gets the SOC flags but stays non-spin-polarised.
        """
        soc_written = qe_defect_setup_from_generator(
            self.defect_gen,
            ecutwfc=self.ecutwfc,
            kpoint_density=self.kpoint_density,
            output_dir=os.path.join(self.tmp, "MgO_defects_soc"),
            soc=True,
            include_bulk=True,
        )
        for name, path in soc_written.items():
            pw = PWin.from_file(path)
            assert pw.system["noncolin"] is True, f"noncolin should be True with SOC (input: {name})"
            assert pw.system["lspinorb"] is True, f"lspinorb should be True with SOC (input: {name})"
            # nspin is not allowed with noncolin (SOC):
            assert "nspin" not in pw.system, f"nspin should be absent with SOC (input: {name})"
            if name == "MgO_bulk":  # neutral bulk reference stays non-spin-polarised
                assert "starting_magnetization" not in pw.system, (
                    f"bulk reference should not be spin-polarised (input: {name})"
                )
            else:
                assert pw.system["starting_magnetization"] == [0.1] * pw.system["ntyp"], (
                    f"starting_magnetization mismatch (input: {name})"
                )

    def test_include_bulk_false(self):
        """``include_bulk=False`` writes only the defect inputs (no bulk reference)."""
        no_bulk_dir = os.path.join(self.tmp, "MgO_defects_no_bulk")
        no_bulk = qe_defect_setup_from_generator(
            self.defect_gen,
            ecutwfc=self.ecutwfc,
            kpoint_density=self.kpoint_density,
            output_dir=no_bulk_dir,
            include_bulk=False,
        )
        assert "MgO_bulk" not in no_bulk
        assert set(no_bulk) == set(self.defect_gen.defect_entries)  # defects only
        assert not os.path.exists(os.path.join(no_bulk_dir, "MgO_bulk"))  # no bulk folder written
        for name, path in no_bulk.items():
            assert os.path.isfile(path), f"missing input file for {name!r}: {path}"


    def test_kpoints_behaviour(self):
        """Reciprocal-density grid, default/explicit shifts, and Γ-only cards."""
        with self.subTest("reciprocal-density grid is a triple of positive ints"):
            grid = _kpoints_grid_from_reciprocal_density(self.mgo_prim, 60)
            assert len(grid) == 3
            assert all(isinstance(k, int) and k > 0 for k in grid)

        with self.subTest("default K_POINTS shift is zero"):
            pw = PWin.from_file(self.defect_written["MgO_bulk"])
            assert [int(s) for s in pw.k_points.shift] == [0, 0, 0]

        with self.subTest("kpoints_shift=(1, 1, 1) is written as the K_POINTS offset"):
            shifted_path = qe_relax_setup_from_structure(
                self.mgo_prim,
                ecutwfc=self.ecutwfc,
                kpoint_density=self.kpoint_density,
                output_dir=os.path.join(self.tmp, "MgO_shifted"),
                kpoints_shift=(1, 1, 1),
            )
            shifted_pw = PWin.from_file(shifted_path)
            assert str(shifted_pw.k_points.option) == "automatic"
            assert [int(s) for s in shifted_pw.k_points.shift] == [1, 1, 1]

        with self.subTest("kpoints=None writes a Γ-only K_POINTS gamma card"):
            gamma_path = os.path.join(self.tmp, "gamma", "pw.in")
            _write_qe_pw_input(
                gamma_path,
                self.mgo_prim,
                {"control": {"calculation": "scf"}, "system": {"ecutwfc": 40}, "electrons": {}},
                None,
            )
            assert "K_POINTS" in open(gamma_path).read()
            assert str(PWin.from_file(gamma_path).k_points.option) == "gamma"

    def test_build_qe_base_settings(self):
        """``_build_qe_base_settings`` sets ibrav/nat/ntyp/pseudo_dir and metal smearing."""
        base = _build_qe_base_settings(
            self.mgo_prim,
            pseudo_dir="./my_pseudos/",
            is_metal=True,
            user_control_settings={"nstep": 200},
            user_system_settings=None,
            user_electron_settings=None,
        )
        assert base["system"]["ibrav"] == 0
        assert base["system"]["nat"] == len(self.mgo_prim)
        assert base["system"]["ntyp"] == 2
        assert base["control"]["pseudo_dir"] == "./my_pseudos/"
        assert base["control"]["nstep"] == 200  # user override merged
        # is_metal -> smearing defaults
        assert base["system"]["occupations"] == "smearing"
        assert base["system"]["smearing"] == "gaussian"
        assert base["system"]["degauss"] == 0.005

    def test_oxidation_states_stripped(self):
        """Oxidation states are removed so ATOMIC_SPECIES uses plain element symbols."""
        oxi_struct = self.mgo_prim.copy()
        oxi_struct.add_oxidation_state_by_element({"Mg": 2, "O": -2})
        path = os.path.join(self.tmp, "oxi", "pw.in")
        _write_qe_pw_input(
            path,
            oxi_struct,
            {"control": {"calculation": "scf"}, "system": {"ecutwfc": 40}, "electrons": {}},
            [4, 4, 4],
        )
        symbols = PWin.from_file(path).atomic_species.symbols
        assert set(symbols) == {"Mg", "O"}  # not "Mg2+"/"O2-"

    def test_invalid_namelist_warns(self):
        """An unrecognised namelist key triggers a warning (and is ignored)."""
        path = os.path.join(self.tmp, "bad_namelist", "pw.in")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _write_qe_pw_input(
                path,
                self.mgo_prim,
                {
                    "control": {"calculation": "scf"},
                    "system": {"ecutwfc": 40},
                    "electron": {"conv_thr": 1e-8},  # typo: should be "electrons"
                },
                [4, 4, 4],
            )
        assert any("unrecognised qe namelist" in str(x.message).lower() for x in w)
        assert PWin.from_file(path).electrons is None

    def test_input_files_validate(self):
        """
        The GGA (vc-)relax, HSE, SOC and convergence (scf) inputs all pass
        ``PWin.validate()`` (no unrecognised keys / malformed cards), without
        emitting warnings.
        """
        hse_path = qe_relax_setup_from_structure(
            self.mgo_prim,
            ecutwfc=self.ecutwfc,
            kpoint_density=self.kpoint_density,
            output_dir=os.path.join(self.tmp, "MgO_validate_hse"),
            use_hse=True,
        )
        soc_path = qe_relax_setup_from_structure(
            self.mgo_prim,
            ecutwfc=self.ecutwfc,
            kpoint_density=self.kpoint_density,
            output_dir=os.path.join(self.tmp, "MgO_validate_soc"),
            soc=True,
        )
        paths = {
            "relax": self.relax_path,
            "hse": hse_path,
            "soc": soc_path,
            "ecut_convergence": self.conv_written["ecut_convergence"][0],
            "kpoint_converge": self.conv_written["kpoint_converge"][0],
        }
        for label, path in paths.items():
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                assert PWin.from_file(path).validate() is True, f"{label} input failed validate(): {path}"


if __name__ == "__main__":
    unittest.main()