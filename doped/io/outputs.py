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
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Spin
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
    magnetization: float | np.ndarray | None = None
    noncollinear: bool | None = None
    vbm: float | None = None
    cbm: float | None = None
    band_gap: float | None = None
    planar_averaged_potentials: dict[int, np.ndarray] | None = None
    site_potentials: list[float] | np.ndarray | None = None
    run_metadata: dict[str, Any] = field(default_factory=dict)

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
