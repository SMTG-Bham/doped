"""
Calculator-agnostic container for the outputs of defect / bulk supercell
calculations.

``CalculationOutputs`` defines the data contract between calculator-specific
parsing code (``doped.io.<calculator>.outputs``) and the calculator-agnostic
analysis functions in ``doped`` (defect identification, charge corrections,
eigenvalue / band-edge analyses, thermodynamics...). Each calculator backend
implements a ``get_calculation_outputs()`` function in its
``doped.io.<calculator>.outputs`` module, returning a populated
``CalculationOutputs`` object -- see the "Adding Support for a New Calculator"
docs page for details.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from monty.json import MontyDecoder, MSONable
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Spin
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.util.typing import PathLike


@dataclass
class CalculationOutputs(MSONable):
    """
    Container for the parsed outputs of a single supercell calculation, in
    calculator-agnostic form.

    Only ``structure`` and ``energy`` are required; all other attributes are
    optional (``None`` if unavailable), being only needed for specific
    analyses -- noted per-attribute below. Analysis functions should use
    :meth:`require` to give informative errors when a needed output was not
    parsed / is not supported by the calculator used.

    Attributes:
        structure (Structure):
            Final (relaxed) structure of the calculation supercell.
        energy (float):
            Final total energy in eV.
        calculator (str):
            Name of the calculator (e.g. ``"vasp"``) which produced these
            outputs, matching the corresponding ``doped.io.<calculator>``
            module name.
        directory (PathLike):
            Directory from which these outputs were parsed, if applicable.
        converged_electronic (bool):
            Whether the electronic self-consistency converged. Used for
            parsing sanity checks.
        converged_ionic (bool):
            Whether the ionic relaxation converged. Used for parsing sanity
            checks.
        efermi (float):
            Fermi level in eV. Needed for eigenvalue / shallow defect
            analyses.
        eigenvalues (dict[Spin, np.ndarray]):
            Band eigenvalues and occupancies as ``{spin: array}`` with array
            shape (nkpoints, nbands, 2), where the last axis is (energy in
            eV, occupation) -- i.e. the ``pymatgen`` ``Vasprun.eigenvalues``
            format. Needed for eigenvalue / shallow defect analyses.
        projected_eigenvalues (dict[Spin, np.ndarray]):
            Orbital projections of the band eigenvalues as ``{spin: array}``
            with array shape (nkpoints, nbands, nions, norbitals). Needed for
            eigenvalue / shallow defect analyses.
        projected_magnetisation (np.ndarray):
            Projected magnetisation with shape (nkpoints, nbands, nions,
            norbitals, 3), for non-collinear calculations.
        kpoint_coords (np.ndarray):
            Fractional coordinates of the calculation k-points, shape
            (nkpoints, 3). Needed for eigenvalue analyses.
        kpoint_weights (np.ndarray):
            Weights of the calculation k-points, shape (nkpoints,). Needed
            for eigenvalue analyses.
        nelect (float):
            Total number of electrons in the calculation. Needed to determine
            defect charge states and spin degeneracies.
        charge (float):
            Net charge of the simulation cell, matching the defect
            charge-state sign convention (i.e. positive charge = electrons
            removed; e.g. determined from the difference in electron count
            relative to the neutral cell, with VASP). ``None`` if this could
            not be determined. Needed for automatic defect charge-state
            determination.
        magnetization (float | np.ndarray):
            Total magnetization of the cell. Needed for spin degeneracy
            determination.
        noncollinear (bool):
            Whether the calculation was non-collinear (e.g. with spin-orbit
            coupling). Needed for band-edge / degeneracy analyses.
        vbm (float):
            Valence band maximum eigenvalue in eV (typically from the bulk
            supercell or a separate bulk band-structure calculation). Needed
            for formation energy calculations.
        cbm (float):
            Conduction band minimum eigenvalue in eV.
        band_gap (float):
            Band gap in eV.
        planar_averaged_potentials (dict[int, np.ndarray]):
            Planar-averaged electrostatic potential along each lattice
            vector, as ``{axis index: 1D array}`` (in eV). Needed for
            Freysoldt (FNV) finite-size charge corrections.
        site_potentials (list[float] | np.ndarray):
            Atomic-site electrostatic potentials (in eV), one per site in
            ``structure`` (e.g. core-level potentials from ``OUTCAR`` files
            with VASP). Needed for Kumagai (eFNV) finite-size charge
            corrections.
        run_metadata (dict):
            Calculator-specific calculation settings / metadata (e.g.
            ``INCAR``/``KPOINTS``/``POTCAR`` data for VASP), used for
            bulk/defect calculation compatibility checks.
        raw (dict):
            Calculator-specific parsed objects from this calculation (e.g.
            ``{"vasprun": Vasprun, "procar": Procar, "computed_entry":
            ComputedStructureEntry}`` for VASP), for reuse by
            calculator-specific code without re-parsing. Not included in
            serialised (``as_dict``) output.
    """

    structure: Structure
    energy: float
    calculator: str | None = None
    directory: PathLike | None = None
    converged_electronic: bool | None = None
    converged_ionic: bool | None = None
    efermi: float | None = None
    eigenvalues: dict[Spin, np.ndarray] | None = None
    projected_eigenvalues: dict[Spin, np.ndarray] | None = None
    projected_magnetisation: np.ndarray | None = None
    kpoint_coords: np.ndarray | None = None
    kpoint_weights: np.ndarray | None = None
    nelect: float | None = None
    charge: float | None = None
    magnetization: float | np.ndarray | None = None
    noncollinear: bool | None = None
    vbm: float | None = None
    cbm: float | None = None
    band_gap: float | None = None
    planar_averaged_potentials: dict[int, np.ndarray] | None = None
    site_potentials: list[float] | np.ndarray | None = None
    run_metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def as_dict(self) -> dict:
        """
        MSON-style ``dict`` representation, excluding the in-memory ``raw``
        objects (which are generally large and/or not serialisable).

        Non-string dictionary keys (``Spin`` keys for eigenvalue data, axis
        indices for planar-averaged potentials) are converted to strings for
        JSON compatibility, and restored by :meth:`from_dict`.
        """
        raw, self.raw = self.raw, {}  # detach raw so it is not (expensively) serialised
        try:
            dct = super().as_dict()
        finally:
            self.raw = raw
        dct.pop("raw", None)

        for key in ("eigenvalues", "projected_eigenvalues", "planar_averaged_potentials"):
            if dct.get(key) is not None:  # Spin / axis-index keys -> strings for JSON compatibility
                dct[key] = {str(getattr(k, "value", k)): v for k, v in dct[key].items()}
        return dct

    @classmethod
    def from_dict(cls, d: dict) -> "CalculationOutputs":
        """
        Reconstitute a ``CalculationOutputs`` object from its :meth:`as_dict`
        representation.
        """
        decoder = MontyDecoder()
        decoded = {
            k: decoder.process_decoded(v) for k, v in d.items() if not k.startswith("@") and k != "raw"
        }
        for key in ("eigenvalues", "projected_eigenvalues"):
            if decoded.get(key) is not None:
                decoded[key] = {Spin(int(k)): np.asarray(v) for k, v in decoded[key].items()}
        if decoded.get("planar_averaged_potentials") is not None:
            decoded["planar_averaged_potentials"] = {
                int(k): np.asarray(v) for k, v in decoded["planar_averaged_potentials"].items()
            }
        for key in ("projected_magnetisation", "kpoint_coords", "kpoint_weights"):
            if decoded.get(key) is not None:
                decoded[key] = np.asarray(decoded[key])
        return cls(**decoded)

    def get_computed_entry(self) -> ComputedStructureEntry:
        """
        Get a ``pymatgen`` ``ComputedStructureEntry`` for this calculation.

        Returns the calculator-parsed entry from ``raw["computed_entry"]``
        when available (e.g. from ``Vasprun.get_computed_entry()`` with VASP,
        including calculation parameters), otherwise builds a bare entry from
        ``structure`` and ``energy``.
        """
        if (computed_entry := self.raw.get("computed_entry")) is not None:
            return computed_entry
        return ComputedStructureEntry(self.structure, self.energy)

    def spin_degeneracy(self, charge_state: int | None = None) -> int:
        """
        Get the spin degeneracy (multiplicity) of this calculation.

        The electron count is determined from ``charge_state`` (with
        ``structure``) if provided, else from ``nelect``, and combined with
        ``magnetization`` to give the spin multiplicity (``2S + 1``). If
        ``magnetization`` is ``None`` (e.g. not supported by the calculator
        used), simple singlet (even-electron) / doublet (odd-electron)
        behaviour is assumed -- see
        :func:`~doped.utils.symmetry._spin_degeneracy_from_num_electrons_and_magnetization`.

        Args:
            charge_state (int):
                The net charge of the system, from which the total number of
                electrons can be determined (with ``structure``). If ``None``
                (default), the ``nelect`` output is used instead.

        Returns:
            int: Spin degeneracy of the system.
        """
        from doped.utils.symmetry import (  # avoid circular imports (symmetry imports doped.core)
            _num_electrons_from_charge_state,
            _spin_degeneracy_from_num_electrons_and_magnetization,
        )

        if charge_state is not None:
            num_electrons = _num_electrons_from_charge_state(self.structure, charge_state)
        else:
            self.require("nelect", task="Spin degeneracy determination (without a known charge state)")
            assert self.nelect is not None  # typing (require() ensures this)
            num_electrons = int(self.nelect)
        return _spin_degeneracy_from_num_electrons_and_magnetization(num_electrons, self.magnetization)

    def require(self, *attrs: str, task: str = "this analysis") -> None:
        """
        Raise an informative ``ValueError`` if any of the named attributes are
        ``None``.

        Args:
            *attrs (str): Names of required ``CalculationOutputs`` attributes.
            task (str): Description of the analysis requiring them, for the
                error message.
        """
        if missing := [attr for attr in attrs if getattr(self, attr) is None]:
            calc = f" {self.calculator}" if self.calculator else ""
            raise ValueError(
                f"{task} requires the {missing} calculation output(s), which could not be parsed from "
                f"(or are not supported by) this{calc} calculation"
                + (f" (in {self.directory})." if self.directory else ".")
            )
