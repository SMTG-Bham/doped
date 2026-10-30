"""
Tests for the GPAW interface in ``doped.io.gpaw``.
"""

import contextlib
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
from ase.io import read
from ase.io.trajectory import Trajectory
from pymatgen.core.structure import Structure
from test_utils import data_dir, gpaw_data_dir

from doped.generation import DefectsGenerator
from doped.io import get_calculation_outputs
from doped.io.gpaw.inputs import DefectDictSet, DefectsSet
from doped.io.gpaw.outputs import _find_gpaw_output, get_potentials_from_input
from doped.parsing import DefectParser, DefectsParser


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
        # GPAW parameter objects are converted to their ``todict()`` forms, giving valid scripts (rather
        # than object reprs) and a JSON-serialisable ``DefectsSet`` (written by ``write_files``):
        gpaw = pytest.importorskip("gpaw")
        gpaw_settings = {"mode": gpaw.PW(400), "eigensolver": gpaw.Davidson(3)}
        relax_set = DefectDictSet(self.structure, gpaw_settings=gpaw_settings)
        content = relax_set._generate_script("structure.cif")
        assert "mode=PW(ecut=400" in content
        assert "eigensolver={'name': 'davidson', 'niter': 3}" in content
        compile(content, "relax.py", "exec")  # valid Python

        DefectsSet(DefectsGenerator(self.structure), gpaw_settings=gpaw_settings).write_files(
            output_path=self.output_dir, processes=1
        )
        assert "mode=PW(ecut=400" in Path(self.output_dir, "Cu_bulk", "singlepoint.py").read_text()
        reloaded = DefectsSet.from_json(os.path.join(self.output_dir, "Cu_DefectsSet.json.gz"))
        assert reloaded.gpaw_settings == {key: value.todict() for key, value in gpaw_settings.items()}

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
