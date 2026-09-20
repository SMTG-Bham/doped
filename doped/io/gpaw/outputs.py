"""
Parsing of GPAW defect / bulk supercell calculation outputs.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core.entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor

from doped.core import Defect, DefectEntry
from doped.parsing import defect_from_structures

_GPAW_OUTPUT_PRIORITY = (
    "relaxed.gpw",
    "singlepoint.gpw",
    "final.gpw",
)


def _find_gpaw_output(
    output_path: str | os.PathLike,
    subfolder: str | os.PathLike | None = None,
) -> str:
    """
    Find a GPAW restart file in a calculation directory.

    If multiple restart files are present, standard relaxation, single-point,
    and final-state filenames are preferred in that order.
    """
    calc_path = Path(output_path)
    if calc_path.is_file():
        if calc_path.name.lower().endswith(".gpw"):
            return str(calc_path)
        raise ValueError(f"GPAW output must be a '.gpw' file: {calc_path}")

    if subfolder is not None and subfolder != ".":
        calc_path /= Path(subfolder)

    if not calc_path.is_dir():
        raise FileNotFoundError(f"GPAW calculation directory not found: {calc_path}")

    gpw_files = [
        path for path in calc_path.iterdir() if path.is_file() and path.name.lower().endswith(".gpw")
    ]
    files_by_name = {path.name.lower(): path for path in gpw_files}
    for preferred_name in _GPAW_OUTPUT_PRIORITY:
        if preferred_name in files_by_name:
            return str(files_by_name[preferred_name])

    if len(gpw_files) == 1:
        return str(gpw_files[0])
    if not gpw_files:
        raise FileNotFoundError(f"No '.gpw' file found in: {calc_path}")

    filenames = ", ".join(sorted(path.name for path in gpw_files))
    raise ValueError(
        f"Multiple GPAW output files found in {calc_path}, with no preferred filename: {filenames}"
    )


def _get_site_potentials_from_calc(calc) -> np.ndarray:
    """
    Get the atom-centred electrostatic site potentials from a ``GPAW``
    calculator.

    ``GPAW``'s ``get_atomic_electrostatic_potentials()`` integrates the pseudo
    Hartree potential against each atom's L=0 compensation-charge shape
    function, which is the direct analogue of the average electrostatic
    potential at the core reported by ``VASP`` in the ``OUTCAR``. It is negated
    here to match the sign convention used by ``doped``/``pydefect`` for eFNV
    (Kumagai-Oba) corrections.

    Args:
        calc (GPAW): ``GPAW`` calculator object.

    Returns:
        np.ndarray: Atomic site potentials (in eV), one per atom.
    """
    return -1.0 * np.array(calc.get_atomic_electrostatic_potentials())


def _get_planar_averaged_potential_from_calc(calc) -> dict[str, np.ndarray]:
    """
    Helper to extract planar-averaged potentials from a GPAW calculator.
    """
    v_ext = calc.get_electrostatic_potential()
    planar_averages = {}
    for i in range(3):
        axes = [0, 1, 2]
        axes.remove(i)
        planar_averages[str(i)] = v_ext.mean(axis=tuple(axes))

    return planar_averages


def get_gpaw_site_potentials(
    gpw_file: str | os.PathLike,
) -> np.ndarray:
    """
    Extracts atomic site potentials from a ``GPAW`` ``.gpw`` file.
    """
    from gpaw import GPAW

    gpw_file = _find_gpaw_output(gpw_file)
    calc = GPAW(gpw_file)
    site_potentials = _get_site_potentials_from_calc(calc)

    if hasattr(calc, "close"):
        calc.close()

    if hasattr(calc, "atoms") and calc.atoms:
        calc.atoms.calc = None

    return site_potentials


def get_gpaw_planar_averaged_potential(
    gpw_file: str | os.PathLike,
) -> dict[str, np.ndarray]:
    """
    Extracts planar-averaged potential from a ``GPAW`` ``.gpw`` file.
    """
    from gpaw import GPAW

    gpw_file = _find_gpaw_output(gpw_file)
    calc = GPAW(gpw_file)
    planar_averages = _get_planar_averaged_potential_from_calc(calc)

    if hasattr(calc, "close"):
        calc.close()

    return planar_averages


class GPAWParser:
    """
    Parser for GPAW calculations to interface with doped.

    Note:
        The Kumagai (eFNV) finite-size charge correction is applied by default
        during parsing, as it is generally preferred. However, the standard
        Freysoldt (FNV) correction is also fully supported. If preferred, users
        can manually apply it to the parsed defects using:
        `defect_entry.get_freysoldt_correction()`
    """

    def __init__(
        self,
        gpw_file: str | os.PathLike,
    ):
        """
        Args:
            gpw_file (str): Path to ``GPAW`` ``.gpw`` file.
        """
        from gpaw import GPAW

        self.gpw_file = _find_gpaw_output(gpw_file)
        self.calc = GPAW(self.gpw_file)
        self.atoms = self.calc.get_atoms()
        self.structure = AseAtomsAdaptor.get_structure(self.atoms)
        self.energy = self.calc.get_potential_energy()

        # Pull charge directly from calculation parameters
        try:
            self.charge = self.calc.parameters.get("charge", None)
        except Exception:
            self.charge = None

    def get_computed_structure_entry(self) -> ComputedStructureEntry:
        """
        Returns a ComputedStructureEntry for the calculation.
        """
        return ComputedStructureEntry(self.structure, self.energy)

    def get_computed_entry(self) -> ComputedEntry:
        """
        Returns a ComputedEntry for the calculation.
        """
        return ComputedEntry(self.structure.composition, self.energy)

    def get_site_potentials(self) -> np.ndarray:
        """
        Returns atomic site potentials.
        """
        return _get_site_potentials_from_calc(self.calc)

    def get_locpot_dict(self) -> dict[str, np.ndarray]:
        """
        Returns planar-averaged potential dictionary.
        """
        return _get_planar_averaged_potential_from_calc(self.calc)

    def get_eigenvalue_properties(self) -> tuple:
        """
        Returns (band_gap, cbm, vbm, efermi).
        """
        # TODO: Band edges are taken purely from the Fermi level here (VBM = highest eigenvalue at or below
        # E_F, CBM = lowest above it), with no reference to occupations. That is only safe for a gapped
        # bulk with E_F in the gap, which is all this is currently used for (see ``_get_gpaw_bulk_data``):
        # it gives meaningless edges for a metallic or heavily-smeared bulk, and silently returns
        # ``efermi`` for both edges (i.e. a zero gap) if no eigenvalue falls on one side. It would also be
        # outright wrong if applied to charged defect supercells, where a partially-occupied in-gap state
        # would be reported as a band edge. Replace with an occupation-based determination, as done for
        # VASP in ``doped.utils.eigenvalues.band_edge_properties_from_outputs``. See the GPAW tracking
        # issue.
        # Basic implementation
        efermi = self.calc.get_fermi_level()
        # GPAW can give eigenvalues for each k-point and spin
        # This is a simplification to get VBM/CBM
        energies = []
        for s in range(self.calc.get_number_of_spins()):
            for k in range(len(self.calc.get_ibz_k_points())):
                energies.extend(self.calc.get_eigenvalues(kpt=k, spin=s))

        energies = sorted(energies)
        # Identify VBM and CBM based on efermi
        vbm = max([e for e in energies if e <= efermi]) if any(e <= efermi for e in energies) else efermi
        cbm = min([e for e in energies if e > efermi]) if any(e > efermi for e in energies) else efermi
        band_gap = cbm - vbm

        return band_gap, cbm, vbm, efermi

    def close(self):
        """
        Closes the underlying GPAW calculator.
        """
        if hasattr(self.calc, "close"):
            self.calc.close()

        # Break reference cycle
        if self.atoms:
            self.atoms.calc = None
        self.calc = None
        self.atoms = None


def _get_gpaw_bulk_data(bulk_parser: GPAWParser, bulk_path: str | os.PathLike) -> dict[str, Any]:
    """
    Parse reusable bulk reference data once.
    """
    band_gap, cbm, vbm, efermi = bulk_parser.get_eigenvalue_properties()
    return {
        "bulk_entry": bulk_parser.get_computed_structure_entry(),
        "bulk_site_potentials": bulk_parser.get_site_potentials(),
        "bulk_locpot_dict": bulk_parser.get_locpot_dict(),
        "bulk_path": str(bulk_path),
        "vbm": vbm,
        "band_gap": band_gap,
        "cbm": cbm,
        "efermi": efermi,
    }


def _get_gpaw_defect_entry_from_parsers(
    defect_parser: GPAWParser,
    bulk_parser: GPAWParser,
    defect_path: str | os.PathLike,
    dielectric: float | np.ndarray | None,
    charge_state: int,
    bulk_data: dict[str, Any],
) -> DefectEntry:
    """
    Build a defect entry from already-open GPAW parsers.
    """
    (
        defect,
        defect_site,  # _relaxed_ defect site
        *_,
    ) = defect_from_structures(
        defect_supercell=defect_parser.structure,
        bulk_supercell=bulk_parser.structure,
        return_all_info=True,
        _parameter_order_warn=False,
    )
    assert isinstance(defect, Defect)  # typing

    return DefectEntry(
        defect=defect,
        charge_state=charge_state,
        sc_entry=defect_parser.get_computed_structure_entry(),
        bulk_entry=bulk_data["bulk_entry"],
        sc_defect_frac_coords=defect_site.frac_coords,
        defect_supercell=defect_parser.structure,
        bulk_supercell=bulk_parser.structure,
        defect_supercell_site=defect_site,
        calculation_metadata={
            "bulk_path": bulk_data["bulk_path"],
            "defect_path": str(defect_path),
            "dielectric": dielectric,
            "bulk_site_potentials": bulk_data["bulk_site_potentials"],
            "defect_site_potentials": defect_parser.get_site_potentials(),
            "bulk_locpot_dict": bulk_data["bulk_locpot_dict"],
            "defect_locpot_dict": defect_parser.get_locpot_dict(),
            "vbm": bulk_data["vbm"],
            "band_gap": bulk_data["band_gap"],
            "cbm": bulk_data["cbm"],
            "efermi": bulk_data["efermi"],
        },
    )


def get_gpaw_defect_entry(
    defect_path: str | os.PathLike,
    bulk_path: str | os.PathLike,
    dielectric: float | np.ndarray | None = None,
    charge_state: int = 0,
    bulk_parser: GPAWParser | None = None,
) -> DefectEntry:
    """
    Create a defect entry from GPAW output files or directories.
    """
    defect_parser = GPAWParser(defect_path)
    close_bulk = bulk_parser is None
    if bulk_parser is None:
        bulk_parser = GPAWParser(bulk_path)

    try:
        bulk_data = _get_gpaw_bulk_data(bulk_parser, bulk_path)
        return _get_gpaw_defect_entry_from_parsers(
            defect_parser=defect_parser,
            bulk_parser=bulk_parser,
            defect_path=defect_path,
            dielectric=dielectric,
            charge_state=charge_state,
            bulk_data=bulk_data,
        )
    finally:
        defect_parser.close()
        if close_bulk:
            bulk_parser.close()


class GPAWDefectsParser:
    """
    Class for rapidly parsing multiple GPAW defect supercell calculations.
    """

    def __init__(
        self,
        output_path: str | os.PathLike = ".",
        dielectric: float | np.ndarray | None = None,
        subfolder: str | os.PathLike | None = None,
        bulk_path: str | os.PathLike | None = None,
    ):
        """
        Args:
            output_path (str): Path to directory containing defect folders.
            dielectric (float or matrix): Dielectric constant for corrections.
            subfolder (str): Optional subfolder within each defect folder.
            bulk_path (str): Path to bulk reference folder.

        Attributes:
            defect_dict (dict): Parsed defect entries keyed by calculation folder name.
        """
        self.output_path = str(output_path)
        self.dielectric = dielectric
        self.subfolder = subfolder

        if bulk_path is None:
            # Try to find bulk folder
            folders = [
                f for f in os.listdir(self.output_path) if os.path.isdir(os.path.join(self.output_path, f))
            ]
            bulk_folders = [f for f in folders if "bulk" in f.lower()]
            if not bulk_folders:
                raise ValueError("Could not find bulk folder. Please specify bulk_path.")
            bulk_folder = sorted(bulk_folders, key=lambda name: (name.lower() != "bulk", name))[0]
            self.bulk_path = os.path.join(self.output_path, bulk_folder)
        else:
            bulk_path = os.fspath(bulk_path)
            self.bulk_path = (
                bulk_path if os.path.isabs(bulk_path) else os.path.join(self.output_path, bulk_path)
            )

        self.defect_dict = self._parse_all()

    @staticmethod
    def _get_charge_state(folder: str, parsed_charge: int | None) -> int:
        """
        Use the GPAW charge, falling back to a signed folder-name component.
        """
        if parsed_charge is not None:
            return int(parsed_charge)
        for component in reversed(folder.split("_")):
            if component.startswith(("+", "-")):
                try:
                    return int(component)
                except ValueError:
                    pass
        return 0

    def _parse_all(self) -> dict[str, DefectEntry]:
        """
        Parse all GPAW defect calculations during initialisation.
        """
        defect_dict = {}
        folders = [
            f for f in os.listdir(self.output_path) if os.path.isdir(os.path.join(self.output_path, f))
        ]

        # Exclude bulk folder
        defect_folders = [
            f
            for f in folders
            if os.path.abspath(os.path.join(self.output_path, f)) != os.path.abspath(self.bulk_path)
        ]

        bulk_parser = GPAWParser(self.bulk_path)
        try:
            bulk_data = _get_gpaw_bulk_data(bulk_parser, self.bulk_path)
            for folder in defect_folders:
                defect_dir = os.path.join(self.output_path, folder)
                try:
                    gpw_file = _find_gpaw_output(defect_dir, self.subfolder)
                except FileNotFoundError:
                    continue
                except ValueError as exc:
                    print(f"Failed to parse {folder}: {exc}")
                    continue

                print(f"Parsing {folder}...")
                defect_parser = None
                try:
                    defect_parser = GPAWParser(gpw_file)
                    charge_state = self._get_charge_state(folder, defect_parser.charge)
                    defect_entry = _get_gpaw_defect_entry_from_parsers(
                        defect_parser=defect_parser,
                        bulk_parser=bulk_parser,
                        defect_path=os.path.dirname(gpw_file),
                        dielectric=self.dielectric,
                        charge_state=charge_state,
                        bulk_data=bulk_data,
                    )

                    if self.dielectric is not None and charge_state != 0:
                        try:
                            defect_entry.get_kumagai_correction()
                        except Exception as exc:
                            print(f"Warning: Kumagai correction failed for {folder}: {exc}")

                    defect_entry.name = folder
                    defect_dict[folder] = defect_entry
                except Exception as exc:
                    print(f"Failed to parse {folder}: {exc}")
                finally:
                    if defect_parser is not None:
                        defect_parser.close()
        finally:
            bulk_parser.close()

        return defect_dict
