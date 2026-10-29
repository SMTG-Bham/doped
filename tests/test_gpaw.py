"""
Tests for the GPAW interface in ``doped.io.gpaw``.
"""

import contextlib
import copy
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from ase.io.trajectory import Trajectory
from pymatgen.core.structure import Structure
from test_utils import (
    _run_func_and_capture_stdout_warnings,
    custom_mpl_image_compare,
    data_dir,
    gpaw_data_dir,
)

from doped.generation import DefectsGenerator
from doped.io import get_calculation_outputs
from doped.io.gpaw.inputs import DefectDictSet, DefectsSet
from doped.io.gpaw.outputs import _find_gpaw_output, get_competing_phase_entry, get_potentials_from_input
from doped.io.inputs import _get_bulk_folder_name
from doped.parsing import DefectParser, DefectsParser
from doped.thermodynamics import DefectThermodynamics


class GPAWTest(unittest.TestCase):
    def setUp(self):
        self.output_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.structure = Structure.from_file(os.path.join(data_dir, "Cu_prim_POSCAR"))

    def test_gpaw_defect_dict_set(self):
        relax_set = DefectDictSet(self.structure, charge_state=1)
        relax_set.write_input(self.output_dir)
        assert Structure.from_file(os.path.join(self.output_dir, "structure.cif")).matches(self.structure)
        content = Path(self.output_dir, "relax.py").read_text()
        assert "charge=1" in content
        assert "mode=PW(ecut=400)" in content  # default
        assert "append_trajectory=True" in content  # continues an interrupted relaxation
        compile(content, "relax.py", "exec")  # valid Python

        # rewriting the inputs in a folder with a previous relaxation warns that it will be continued:
        Path(self.output_dir, "relax.traj").touch()
        with pytest.warns(UserWarning, match="relax.traj already exists"):
            relax_set.write_input(self.output_dir)

    def test_gpaw_defect_dict_set_custom(self):
        # including non-string ``xc`` (e.g. dict-form hybrid functional settings) and ``txt`` values:
        for gpaw_settings, expected in (
            ({"kpts": {"size": (2, 2, 2), "gamma": True}}, "kpts={'size': (2, 2, 2), 'gamma': True},"),
            ({"xc": {"name": "HSE06", "backend": "pw"}}, "xc={'name': 'HSE06', 'backend': 'pw'},"),
            ({"txt": None}, "txt=None,"),
            ({"txt": Path("log.txt")}, "txt='log.txt',"),
        ):
            relax_set = DefectDictSet(self.structure, gpaw_settings=gpaw_settings)
            assert expected in relax_set._generate_script("structure.cif")

    def test_gpaw_defect_dict_set_lcao(self):
        gpaw_settings = {"mode": {"name": "lcao", "basis": "dzp"}}
        relax_set = DefectDictSet(self.structure, gpaw_settings=gpaw_settings)
        content = relax_set._generate_script("structure.cif")
        assert "mode=LCAO()" in content
        assert "basis='dzp'" in content
        assert "from gpaw import GPAW, PW, LCAO, FD" in content
        assert gpaw_settings == {"mode": {"name": "lcao", "basis": "dzp"}}  # not modified

    def test_gpaw_invalid_settings_write_nothing(self):
        # invalid settings raise before the output folder is created or any file is written:
        output_path = os.path.join(self.output_dir, "new_folder")
        for gpaw_settings, match in [
            ({"charge": 1}, r"gpaw_settings\['charge'\]"),  # set with ``charge_state`` instead
            ({"mode": {"name": "pw", "basis": "dzp"}, "basis": "sz"}, "Conflicting basis settings"),
            ({"initial_magnetic_moments": [1.0] * (len(self.structure) + 1)}, "one value per atom"),
        ]:
            with pytest.raises(ValueError, match=match):
                DefectDictSet(self.structure, gpaw_settings=gpaw_settings).write_input(output_path)
            assert not os.path.exists(output_path)

    def test_gpaw_singlepoint_set(self):
        singlepoint_set = DefectDictSet(self.structure, calculation_type="singlepoint")
        content = singlepoint_set._generate_script("structure.cif")
        assert "calc.write('singlepoint.gpw')" in content
        assert "dyn.run" not in content  # no relaxation
        compile(content, "singlepoint.py", "exec")  # valid Python

    def test_gpaw_initial_magnetic_moments(self):
        input_set = DefectDictSet(self.structure, gpaw_settings={"initial_magnetic_moments": [1.0]})
        content = input_set._generate_script("structure.cif")
        assert "atoms.set_initial_magnetic_moments([1.0])" in content  # (Cu_prim_POSCAR has one site)
        assert "initial_magnetic_moments=" not in content  # not passed to ``GPAW()``

        # per-element moments are expanded for the structure, with unlisted elements set to zero:
        structure = self.structure * (2, 1, 1)
        structure.replace(0, "Ni")
        input_set = DefectDictSet(structure, gpaw_settings={"initial_magnetic_moments": {"Ni": 2.0}})
        content = input_set._generate_script("structure.cif")
        assert "atoms.set_initial_magnetic_moments([2.0, 0.0])" in content

    def test_gpaw_defects_set(self):
        defect_gen = DefectsGenerator(self.structure)
        with pytest.raises(TypeError, match="unexpected_setting"):  # GPAW settings go in gpaw_settings
            DefectsSet(defect_gen, unexpected_setting=True)
        with pytest.raises(ValueError, match="dict with `DefectsSet`"):
            DefectsSet(defect_gen, gpaw_settings={"initial_magnetic_moments": [1.0]})

        DefectsSet(
            defect_gen,
            gpaw_settings={"mode": {"name": "pw", "ecut": 500}, "initial_magnetic_moments": {"Cu": 1.0}},
        ).write_files(output_path=self.output_dir, processes=1)
        bulk_structure = Structure.from_file(os.path.join(self.output_dir, "Cu_bulk", "structure.cif"))
        assert bulk_structure.matches(defect_gen.bulk_supercell)

        # one (doped-named) folder per defect, plus the neutral bulk reference (static, regardless of the
        # defects' ``calculation_type``), with the settings forwarded and per-element moments expanded for
        # each supercell (which differ in size):
        scripts = {name: ("relax.py", e.charge_state) for name, e in defect_gen.defect_entries.items()}
        for folder, (script_name, charge) in (scripts | {"Cu_bulk": ("singlepoint.py", 0)}).items():
            folder_path = Path(self.output_dir, folder)
            assert sorted(os.listdir(folder_path)) == [script_name, "structure.cif"]
            script = (folder_path / script_name).read_text()
            n_atoms = len(Structure.from_file(folder_path / "structure.cif"))
            assert "mode=PW(ecut=500)" in script
            assert f"charge={charge}" in script
            assert f"atoms.set_initial_magnetic_moments({[1.0] * n_atoms!r})" in script

    def test_gpaw_parameter_objects(self):
        # GPAW parameter objects are converted to their ``todict()`` forms, giving runnable scripts (rather
        # than object reprs; see ``test_gpaw_end_to_end`` for ``DefectsSet``, and its JSON serialisation):
        gpaw = pytest.importorskip("gpaw")
        gpaw_settings = {"mode": gpaw.PW(400), "eigensolver": gpaw.Davidson(3)}
        DefectDictSet(self.structure, gpaw_settings=gpaw_settings).write_input(self.output_dir)
        content = Path(self.output_dir, "relax.py").read_text()
        assert "mode=PW(ecut=400" in content
        assert "eigensolver={'name': 'davidson', 'niter': 3}" in content
        _run_script_setup(Path(self.output_dir, "relax.py"))  # runs, with GPAW input validation

    def test_find_gpaw_output(self):
        calc_dir = os.path.join(self.output_dir, "calculation")
        os.makedirs(calc_dir)

        custom_output = os.path.join(calc_dir, "custom.gpw")
        Path(custom_output).touch()
        assert _find_gpaw_output(calc_dir) == custom_output

        relaxed_output = os.path.join(calc_dir, "relaxed.gpw")
        Path(relaxed_output).touch()
        assert _find_gpaw_output(calc_dir) == relaxed_output
        assert _find_gpaw_output(custom_output) == custom_output

    def test_gpaw_defects_parser(self):
        """
        Parse the MgO GPAW calculations with ``DefectsParser`` (structure-based
        naming, eFNV corrections and calculation metadata, with
        ``calculator="gpaw"`` so that later potential loading, e.g. for the FNV
        correction, dispatches to ``doped.io.gpaw.outputs``), cross-checked
        with the single-defect ``DefectParser``.
        """
        pytest.importorskip("gpaw")
        mgo_dir = os.path.join(gpaw_data_dir, "MgO")
        bulk_dir = os.path.join(mgo_dir, "bulk")
        dp = DefectsParser(  # default ``processes``, so this also covers the multiprocessing path
            output_path=mgo_dir,
            bulk_path=bulk_dir,
            dielectric=8.8963,
            calculator="gpaw",
            json_filename=False,  # don't write a parsed-dict JSON into the test data directory
        )
        expected_corrections = {  # names from structure analysis, so the unrelaxed folder doesn't collide
            "v_Mg_+1": -0.05491517,
            "v_Mg_-2": 1.20301268,
            "Mg_O_+1": 0.36016471,
            "Mg_O_unrelaxed_+1": 0.39874949,
        }
        corrections = {k: e.corrections["kumagai_charge_correction"] for k, e in dp.defect_dict.items()}
        assert corrections == pytest.approx(expected_corrections, abs=1e-3)
        for entry in dp.defect_dict.values():
            assert entry.calculation_metadata["calculator"] == "gpaw"
            run_metadata = entry.calculation_metadata["run_metadata"]
            assert run_metadata["defect_gpaw_parameters"]["mode"]["name"] == "pw"
            assert run_metadata["bulk_gpaw_parameters"].get("charge", 0) == 0  # GPAW >= 26 omits defaults
            # the bulk & defect calculations here differ only in charge, which is excluded:
            assert entry.calculation_metadata["mismatching_gpaw_parameters"] is False
            assert entry.calculation_metadata["relaxed point symmetry"]

        # the single-defect parser (which parses the bulk itself) gives the same correction:
        entry = DefectParser.from_paths(
            os.path.join(mgo_dir, "Mg_O_unrelaxed"), bulk_dir, dielectric=8.8963, calculator="gpaw"
        ).defect_entry
        assert entry.corrections["kumagai_charge_correction"] == pytest.approx(0.39874949, abs=1e-3)

        # FNV (eFNV is used by default when parsing), with the planar-averaged potentials loaded from the
        # calculation folders (only physically valid for ``v_Mg_-2`` here, as ``v_Mg_+1`` and ``Mg_O_+1``
        # have delocalised carriers, but all tracked as regression values):
        expected_fnv = {"v_Mg_+1": -0.10899449, "v_Mg_-2": 1.27743123, "Mg_O_+1": 0.12075791}
        for name, expected in expected_fnv.items():
            entry = dp.defect_dict[name]
            entry.corrections.pop("kumagai_charge_correction")  # else warns of double-counting with eFNV
            entry.get_freysoldt_correction(
                defect_planar_averaged_potentials=entry.calculation_metadata["defect_path"],
                bulk_planar_averaged_potentials=bulk_dir,
            )
            assert entry.corrections["freysoldt_charge_correction"] == pytest.approx(expected, abs=1e-3)

    def test_gpaw_calculation_outputs(self):
        """
        Test ``get_calculation_outputs()``, the ``doped.io`` backend protocol
        entry point through which ``doped``'s generic machinery parses GPAW
        calculations, and the lazy loading of potentials from the calculation
        folder.
        """
        pytest.importorskip("gpaw")
        bulk_dir = os.path.join(gpaw_data_dir, "MgO", "bulk")
        bulk_outputs = get_calculation_outputs(bulk_dir, calculator="gpaw", label="bulk")
        assert bulk_outputs.calculator == "gpaw"
        assert bulk_outputs.charge == 0
        assert len(bulk_outputs.structure) == len(bulk_outputs.site_potentials)
        assert set(bulk_outputs.planar_averaged_potentials) == {0, 1, 2}
        assert bulk_outputs.converged_electronic
        assert bulk_outputs.nelect > 0
        assert bulk_outputs.band_gap > 0  # band edges are only taken for the (neutral) bulk
        assert bulk_outputs.vbm < bulk_outputs.efermi < bulk_outputs.cbm
        assert "gpaw_parameters" in bulk_outputs.run_metadata
        # eigenvalues are (n_kpoints, n_bands, 2), i.e. energy and occupancy, as pymatgen expects:
        assert all(eigs.ndim == 3 and eigs.shape[-1] == 2 for eigs in bulk_outputs.eigenvalues.values())
        assert len(bulk_outputs.kpoint_weights) == len(bulk_outputs.kpoint_coords)

        defect_outputs = get_calculation_outputs(
            os.path.join(gpaw_data_dir, "MgO", "v_Mg_-2"), calculator="gpaw", label="defect"
        )
        assert defect_outputs.charge == -2
        assert defect_outputs.band_gap is None  # Fermi-level band edges are meaningless when charged
        assert len(defect_outputs.structure) == len(bulk_outputs.structure) - 1  # a vacancy

        # the potentials loaded lazily from the folder match (with string axis keys):
        np.testing.assert_allclose(
            get_potentials_from_input(bulk_dir, "site"), bulk_outputs.site_potentials
        )
        planar_potentials = get_potentials_from_input(bulk_dir)
        assert {int(axis) for axis in planar_potentials} == set(bulk_outputs.planar_averaged_potentials)
        for axis, potentials in planar_potentials.items():
            np.testing.assert_allclose(potentials, bulk_outputs.planar_averaged_potentials[int(axis)])
        # already-parsed potentials are returned as-is, and anything else is rejected:
        assert get_potentials_from_input(planar_potentials) is planar_potentials
        with pytest.raises(TypeError, match=r"bulk potentials input must be either a path"):
            get_potentials_from_input(12345, dir_type="bulk")

    def test_gpaw_graphene_2d_handling(self):
        """
        Test eFNV corrections for vacancy, interstitial and substitution
        defects in a highly anisotropic 2D supercell (graphene), with a
        metallic in- plane and vacuum-like out-of-plane dielectric response.
        """
        pytest.importorskip("gpaw")
        dp = DefectsParser(
            output_path=os.path.join(gpaw_data_dir, "Graphene"),
            dielectric=np.diag([1e6, 1e6, 1.0]),
            calculator="gpaw",
            json_filename=False,
        )
        corrections = {k: e.corrections["kumagai_charge_correction"] for k, e in dp.defect_dict.items()}
        # defect region radius reduced to sample the 3 furthest sites (2D slab), with Ewald sums evaluated
        # in the eps^-1/2-scaled frame (converged w.r.t. ``accuracy``, unlike in the original frame here):
        expected_corrections = {"v_C_+1": -0.88637699, "C_i_C3v_+4": -41.31807622, "N_C_-2": 5.56857059}
        assert corrections == pytest.approx(expected_corrections, abs=1e-3)


def _run_script_setup(script_path: Path) -> dict:
    """
    Run a generated GPAW script up to its calculation (reading the structure
    and setting up the calculator, with real GPAW input validation), in its
    folder, and return the script namespace.
    """
    namespace: dict = {}
    with contextlib.chdir(script_path.parent):
        exec(script_path.read_text().split("atoms.calc = calc")[0], namespace)
    return namespace


@pytest.mark.parametrize(
    "gpaw_settings",
    [
        {"mode": {"name": "lcao", "basis": "dzp"}},
        {"mode": {"name": "lcao"}, "basis": {"Cu": "dzp"}},
        {"mode": "lcao", "basis": "dzp"},
        {"mode": {"name": "lcao", "basis": "dzp"}, "basis": "dzp"},
        {"mode": {"name": "pw", "basis": "dzp"}},
    ],
)
def test_basis_script_initializes_gpaw(gpaw_settings, tmp_path):
    """
    Set up the GPAW calculator from the generated script, with real GPAW input
    validation.
    """
    pytest.importorskip("gpaw")
    structure = Structure.from_file(os.path.join(data_dir, "Cu_prim_POSCAR"))
    DefectDictSet(structure, gpaw_settings=gpaw_settings, calculation_type="singlepoint").write_input(
        tmp_path
    )
    basis = _run_script_setup(tmp_path / "singlepoint.py")["calc"].parameters.basis
    expected = gpaw_settings.get("basis", "dzp")
    assert basis in (expected, {"default": expected})  # GPAW >= 26 normalises a string basis to this


def test_relax_script_continues_from_trajectory(tmp_path):
    """
    ``relax.py`` starts from the last frame of ``relax.traj`` if present (e.g.
    when resubmitting a timed-out relaxation), otherwise ``structure.cif``.
    """
    pytest.importorskip("gpaw")
    DefectDictSet(Structure.from_file(os.path.join(data_dir, "Cu_prim_POSCAR"))).write_input(tmp_path)
    atoms = read(tmp_path / "structure.cif")
    moved = atoms.copy()
    moved.positions += 0.1

    # no trajectory, an empty or header-only one (as left by a run interrupted before its first step, with
    # ASE < 3.27), then a previous relaxation:
    for frames, expected in ((None, atoms), (b"", atoms), ([], atoms), ([atoms, moved], moved)):
        if isinstance(frames, bytes):
            (tmp_path / "relax.traj").write_bytes(frames)
        elif frames is not None:
            with Trajectory(tmp_path / "relax.traj", "w") as trajectory:
                for frame in frames:
                    trajectory.write(frame)

        positions = _run_script_setup(tmp_path / "relax.py")["atoms"].positions
        np.testing.assert_allclose(positions, expected.positions)


def _gpaw_defect_workflow(tmp_path: Path, supercell: Structure, names: tuple[str, ...], **kwargs):
    """
    Generate, write, run and parse GPAW calculations for the named defects in
    ``supercell`` (spin-paired, at Gamma, for speed), with GPAW parameter
    objects in ``gpaw_settings`` (converted to their ``todict()`` forms for the
    generated scripts and the ``DefectsSet`` JSON). ``kwargs`` are passed to
    ``DefectsParser``.

    Returns the ``DefectsParser``, and the warnings from defect generation and
    parsing.
    """
    gpaw = pytest.importorskip("gpaw")
    gpaw_settings = {"mode": gpaw.PW(300), "eigensolver": gpaw.Davidson(3), "spinpol": False}
    defect_gen, _, generation_warnings = _run_func_and_capture_stdout_warnings(
        DefectsGenerator, supercell, generate_supercell=False
    )
    defects_set = DefectsSet([defect_gen[name] for name in names], gpaw_settings=gpaw_settings)
    defects_set.write_files(tmp_path)
    stored = DefectsSet.from_json(tmp_path / defects_set.json_name).gpaw_settings
    assert stored == gpaw_settings | {k: gpaw_settings[k].todict() for k in ("mode", "eigensolver")}

    bulk_folder = _get_bulk_folder_name(defect_gen.bulk_supercell)
    for folder, script in [(name, "relax.py") for name in names] + [(bulk_folder, "singlepoint.py")]:
        with contextlib.chdir(tmp_path / folder):
            runpy.run_path(script)

    dp, _, parsing_warnings = _run_func_and_capture_stdout_warnings(  # finds the bulk folder itself
        DefectsParser, output_path=tmp_path, calculator="gpaw", **kwargs
    )
    assert set(dp.defect_dict) == set(names)
    for entry in dp.defect_dict.values():
        assert entry.calculation_metadata["mismatching_gpaw_parameters"] is False
    return dp, generation_warnings + parsing_warnings


def _check_warnings(warnings: list, expected: list[tuple[str, ...]]):
    """
    Check that the ``UserWarning`` messages in ``warnings`` comprise exactly
    one warning matching each tuple of substrings in ``expected``.
    """
    messages = [str(warning.message) for warning in warnings if issubclass(warning.category, UserWarning)]
    assert len(messages) == len(expected), messages
    for substrings in expected:
        assert sum(all(s in message for s in substrings) for message in messages) == 1, substrings


class GPAWCuEndToEndTest(unittest.TestCase):
    """
    End-to-end GPAW defect workflow (``_gpaw_defect_workflow``, run once for
    the class) for neutral defects in an 8-atom Cu supercell (metallic, so not
    charged).

    Includes an octahedral interstitial and a vacancy (which does not relax due
    to symmetry, with each other atom sitting midway between the vacancy and
    one of its periodic images).
    """

    @classmethod
    def setUpClass(cls):
        supercell = Structure.from_file(os.path.join(data_dir, "Cu_prim_POSCAR")) * 2
        tmp_path = Path(cls.enterClassContext(tempfile.TemporaryDirectory()))
        cls.names = ("v_Cu_0", "Cu_i_Oh_0")
        cls.dp, cls.warnings = _gpaw_defect_workflow(tmp_path, supercell, cls.names)
        cls.entries = [cls.dp.defect_dict[name] for name in cls.names]
        cls.chempots = {"Cu": cls.entries[0].bulk_entry.energy_per_atom}  # mu_Cu = bulk energy per atom

    def test_warnings(self):
        _check_warnings(  # (small supercell warnings)
            self.warnings,
            [
                ("minimum image distance = 5.06",),
                ("parsing v_Cu_0", "first coordination shell distance ~2.53"),
                ("parsing Cu_i_Oh_0", "first coordination shell distance ~2.01"),
                ("'v_Cu_0'", "'Cu_i_Oh_0'", "each encountered", "more than 10% of the 8 sites"),
            ],
        )

    def test_formation_energies(self):  # ~1 eV for the v_Cu, 6 eV for Cu_i, as expected for Cu
        formation_energies = [entry.formation_energy(chempots=self.chempots) for entry in self.entries]
        assert formation_energies == pytest.approx([0.989, 6.017], abs=0.01)

    def test_geometries(self):
        # unrelaxed vacancy (12 nearest neighbours at a/sqrt(2)), and symmetric outward relaxation of the
        # interstitial's 6 nearest neighbours (from a/2 = 1.79 Å):
        for entry, distances in zip(self.entries, ([2.53] * 12, [2.014] * 6), strict=True):
            neighbours = entry.defect_supercell.get_neighbors(entry.defect_supercell_site, 3)
            assert sorted(n.nn_distance for n in neighbours) == pytest.approx(distances, abs=0.01)
        assert self.entries[1].calculation_metadata["relaxed point symmetry"] == "Oh"


class GPAWMgOEndToEndTest(unittest.TestCase):
    """
    End-to-end GPAW defect workflow (``_gpaw_defect_workflow``, run once for
    the class) for charged defects (with eFNV corrections) in insulating MgO
    (3.65 eV band gap with these settings), in the 8-atom conventional cell.

    Includes neutral and charged vacancies (which, as in Cu, do not relax due
    to symmetry), and a charged interstitial (``v_O_+2`` is avoided, as its
    empty defect level is resonant with the valence band in this small cell).
    """

    dp: DefectsParser

    @classmethod
    def setUpClass(cls):
        supercell = Structure.from_spacegroup("Fm-3m", np.eye(3) * 4.21, ["Mg", "O"], [[0] * 3, [0.5] * 3])
        tmp_path = Path(cls.enterClassContext(tempfile.TemporaryDirectory()))
        cls.names = ("v_O_0", "v_Mg_0", "v_Mg_-2", "Mg_i_Td_+2")
        cls.dp, cls.warnings = _gpaw_defect_workflow(tmp_path, supercell, cls.names, dielectric=8.8963)
        cls.entries = [cls.dp.defect_dict[name] for name in cls.names]
        # mu_Mg + mu_O = E(MgO), at a moderately O-rich point within the MgO stability region for this
        # Gamma-only bulk energy (mu_O and mu_Mg 0.66 and 3.29 eV below the O2 and Mg metal energies from
        # separate GPAW calculations at these settings):
        mu = cls.entries[0].bulk_entry.energy_per_atom
        cls.chempots = {"Mg": mu - 3, "O": mu + 3}

    def test_warnings(self):
        _check_warnings(  # (small supercell warnings)
            self.warnings,
            [
                ("minimum image distance = 4.21",),
                (
                    "parsing Mg_i_Td_+2",
                    "first coordination shell distance ~1.79",
                    "Fewer than 3 atomic sites lie outside the eFNV defect region radius",
                ),
                ("'v_O_0'", "'v_Mg_0'", "'v_Mg_-2'", "each encountered", "shell distance ~2.10"),
                ("'v_O_0'", "'v_Mg_0'", "'v_Mg_-2'", "'Mg_i_Td_+2'", "each", "more than 10% of the 8"),
                ("Estimated error in the Kumagai (eFNV) charge correction", "v_Mg_-2: "),
            ],
        )

    def test_band_gap_corrections_formation_energies(self):
        assert self.entries[0].calculation_metadata["band_gap"] == pytest.approx(3.654, abs=0.01)
        corrections = [entry.corrections.get("kumagai_charge_correction", 0) for entry in self.entries]
        assert corrections == pytest.approx([0, 0, 7.324, 4.830], abs=0.01)
        formation_energies = [entry.formation_energy(chempots=self.chempots) for entry in self.entries]
        assert formation_energies == pytest.approx([0.459, 2.558, 7.690, 8.375], abs=0.01)  # at the VBM

    def test_geometries(self):
        # unrelaxed vacancies (6 nearest neighbours at a/2), and Td-symmetric relaxation around the
        # interstitial (4 O neighbours inwards and 4 Mg outwards, from sqrt(3)a/4 = 1.82 Å):
        expected_distances = [[2.105] * 6] * 3 + [[1.791] * 4 + [2.095] * 4]
        for entry, distances in zip(self.entries, expected_distances, strict=True):
            neighbours = entry.defect_supercell.get_neighbors(entry.defect_supercell_site, 2.5)
            assert sorted(n.nn_distance for n in neighbours) == pytest.approx(distances, abs=0.01)
        assert self.entries[-1].calculation_metadata["relaxed point symmetry"] == "Td"

    @custom_mpl_image_compare("GPAW_MgO_formation_energies.png")
    def test_formation_energy_plot(self):
        thermo = DefectThermodynamics(self.entries, chempots=self.chempots)
        ((transition_level, charges),) = thermo.transition_level_map["v_Mg"].items()
        assert (transition_level, charges) == (pytest.approx(2.566, abs=0.01), [0, -2])  # in the gap
        fig, _, warnings = _run_func_and_capture_stdout_warnings(thermo.plot)
        _check_warnings(warnings, [])
        return fig

    def _correction_plot(self, name: str, fnv: bool = False):
        """
        Plot the eFNV (default) or FNV correction for the named defect, using a
        copy of its entry (as the correction is stored in its ``corrections``),
        with the FNV planar-averaged potentials loaded from the calculation
        folders.

        Returns the figure and the warnings (here, estimated correction errors
        above ``error_tolerance``, or the eFNV sampling region being reduced,
        due to the small supercell).
        """
        entry = copy.deepcopy(self.dp.defect_dict[name])
        kwargs: dict = {}
        if fnv:
            entry.corrections.pop("kumagai_charge_correction")  # else warns of double-counting with eFNV
            kwargs = {
                f"{label}_planar_averaged_potentials": entry.calculation_metadata[f"{label}_path"]
                for label in ("defect", "bulk")
            }
        get_correction = entry.get_freysoldt_correction if fnv else entry.get_kumagai_correction
        (_corr, fig), _, w = _run_func_and_capture_stdout_warnings(get_correction, plot=True, **kwargs)
        return fig, w

    @custom_mpl_image_compare("GPAW_MgO_v_Mg_-2_eFNV_plot.png")
    def test_efnv_plot_vacancy(self):
        fig, warnings = self._correction_plot("v_Mg_-2")
        _check_warnings(warnings, [("(eFNV) charge correction for defect v_Mg_-2 is",)])
        return fig

    @custom_mpl_image_compare("GPAW_MgO_Mg_i_Td_+2_eFNV_plot.png")
    def test_efnv_plot_interstitial(self):
        fig, warnings = self._correction_plot("Mg_i_Td_+2")
        _check_warnings(warnings, [("Fewer than 3 atomic sites lie outside the eFNV defect region",)])
        return fig

    @custom_mpl_image_compare("GPAW_MgO_v_Mg_-2_FNV_plot.png")
    def test_fnv_plot_vacancy(self):
        fig, warnings = self._correction_plot("v_Mg_-2", fnv=True)
        _check_warnings(warnings, [("(FNV) charge correction for defect v_Mg_-2 is",)])
        return fig

    @custom_mpl_image_compare("GPAW_MgO_Mg_i_Td_+2_FNV_plot.png")
    def test_fnv_plot_interstitial(self):
        fig, warnings = self._correction_plot("Mg_i_Td_+2", fnv=True)
        _check_warnings(warnings, [("(FNV) charge correction for defect Mg_i_Td_+2 is",)])
        return fig


def test_competing_phase_entry_real_gpaw():
    """
    Read the shipped neutral restart without loading defect potentials.
    """
    pytest.importorskip("gpaw")
    from monty.json import MontyDecoder, MontyEncoder

    path = Path(gpaw_data_dir) / "MgO" / "bulk"
    entry = get_competing_phase_entry(path)
    assert entry.composition.reduced_formula == "MgO"
    assert entry.energy == pytest.approx(73.25088079545509, abs=1e-6)
    assert entry.data["folder"] == str(path)
    assert entry.data["nsites"] == len(entry.structure)
    assert entry.data["gpaw_parameters"]["mode"]["name"] == "pw"
    assert "converged_ionic" not in entry.data
    # The entry contains no live GPAW/MPI objects and can be serialised for later analysis.
    roundtrip = MontyDecoder().decode(MontyEncoder().encode(entry))
    assert roundtrip.energy == entry.energy
    assert roundtrip.structure == entry.structure


@pytest.mark.parametrize("new_parameters", [False, True])
def test_competing_phase_analyzer_gpaw(tmp_path, monkeypatch, new_parameters):
    """
    Check discovery, elemental references and chemical potentials with
    synthetic GPAW outputs.
    """
    gpaw = pytest.importorskip("gpaw")
    from doped.chemical_potentials import CompetingPhasesAnalyzer

    atoms_and_energies = {
        "MgO": (Atoms("MgO", positions=[[0, 0, 0], [2, 2, 2]], cell=[4, 4, 4], pbc=True), -12.0),
        "Mg": (Atoms("Mg", cell=[3, 3, 3], pbc=True), -1.0),
        "O2": (Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.2]], cell=[12, 12, 12], pbc=True), -6.0),
    }
    opened = []

    class PhaseCalculator:
        def __init__(self, filename, **kwargs):
            phase = Path(filename).parent.name
            self.atoms, self.energy = atoms_and_energies[phase]
            parameters = {"mode": {"name": "pw", "ecut": 400}, "xc": "PBE"}
            self.parameters = SimpleNamespace(todict=lambda: parameters) if new_parameters else parameters
            self.scf = SimpleNamespace(converged=True)
            self.closed = False
            opened.append(self)

        def get_atoms(self):
            return self.atoms.copy()

        def get_potential_energy(self):
            return self.energy

        def get_magnetic_moment(self):
            return 2.0 if self.atoms.get_chemical_formula() == "O2" else 0.0

        def close(self):
            self.closed = True

    monkeypatch.setattr(gpaw, "GPAW", PhaseCalculator)
    for phase in atoms_and_energies:
        folder = tmp_path / phase
        folder.mkdir()
        (folder / "singlepoint.gpw").touch()

    analyzer = CompetingPhasesAnalyzer("MgO", entries=tmp_path, calculator="gpaw", processes=1)
    assert len(analyzer.entries) == 3
    assert analyzer.chempots["elemental_refs"] == {"Mg": -1.0, "O": -3.0}
    limits = list(analyzer.chempots["limits_wrt_el_refs"].values())
    np.testing.assert_allclose(sorted((mu["Mg"], mu["O"]) for mu in limits), [(-8, 0), (0, -8)])
    assert set(analyzer.parsed_folders) == {str(tmp_path / phase) for phase in atoms_and_energies}
    assert all(calc.closed for calc in opened)
    assert len(analyzer.get_formation_energy_df()) == 3


def test_competing_phase_entry_rejects_charge_and_closes(tmp_path, monkeypatch):
    """
    Do not use charged defect energies as neutral competing phases; close on
    errors too.
    """
    gpaw = pytest.importorskip("gpaw")
    path = tmp_path / "singlepoint.gpw"
    path.touch()
    calc = SimpleNamespace(parameters={"charge": 1}, atoms=Atoms("H"))
    closed = []
    calc.close = lambda: closed.append(True)
    monkeypatch.setattr(gpaw, "GPAW", lambda *args, **kwargs: calc)
    with pytest.raises(ValueError, match="must be neutral"):
        get_competing_phase_entry(path)
    assert closed == [True]
    assert calc.atoms.calc is None
