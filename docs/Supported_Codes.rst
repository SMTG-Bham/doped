.. _supported-codes:

Supported Codes
===============

Most of ``doped`` (defect generation, site & symmetry analysis, finite-size
charge corrections, thermodynamics, chemical potentials...) is independent of
the code used for the underlying (DFT) calculations. The code-specific parts
-- writing calculation inputs and parsing calculation outputs -- are in the
``doped.io.<code>`` subpackages (see :doc:`doped.io`), which are chosen with:

- **Input generation:** The ``DefectsSet`` class of the corresponding
  ``inputs`` module (e.g. :class:`doped.io.vasp.inputs.DefectsSet`) for defect
  supercell calculations, and the ``calculator`` argument of |CompetingPhases|
  for competing phase calculations.
- **Output parsing:** The ``calculator`` argument of |DefectsParser| and
  |CompetingPhasesAnalyzer|.

``VASP`` is the default (``calculator="vasp"``) and most extensively tested
code. This page summarises the key differences in the supported functionality
and default settings between codes. To add support for another code, see
:doc:`Adding_Calculators`.

.. note::

    The default settings are often reasonable starting points which `roughly`
    match typical values needed for accurate defect calculations, but should
    always be checked (and converged) for your system. Whichever code is
    used, the settings which affect energies (functional, basis set / energy
    cutoff, pseudopotentials...) must be consistent between the defect, bulk
    and competing phase calculations – this is checked automatically when parsing.

Functionality
-------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * -
     - ``VASP``
     - ``GPAW`` (experimental)
   * - ``calculator``
     - ``"vasp"`` (default)
     - ``"gpaw"``
   * - Defect supercell inputs
     - ✓ (:class:`doped.io.vasp.inputs.DefectsSet`)
     - ✓ (:class:`doped.io.gpaw.inputs.DefectsSet`)
   * - Defect calculation stages
     - ``vasp_gam`` (Γ-point only), ``vasp_nkred_std`` (hybrid DFT with
       ``NKRED``) and ``vasp_std`` relaxations, and ``vasp_ncl`` single-point
       calculations with spin-orbit coupling (SOC), each in its own subfolder
     - One relaxation per defect
   * - Competing phase inputs
     - ✓ (`k`-point convergence tests, relaxations and single-point
       calculations)
     - ✗
   * - Defect & competing phase parsing
     - ✓ (|DefectsParser| & |CompetingPhasesAnalyzer|, from
       ``vasprun.xml(.gz)`` files)
     - ✓ (|DefectsParser| & |CompetingPhasesAnalyzer|, from ``.gpw`` restart
       files)
   * - Finite-size charge corrections
     - Kumagai (eFNV; from ``OUTCAR(.gz)`` files) and Freysoldt (FNV; from
       ``LOCPOT(.gz)`` files)
     - Kumagai (eFNV) and Freysoldt (FNV), both from the ``.gpw`` files
   * - Eigenvalue & orbital projection analysis
     - ✓ (projections from ``vasprun.xml(.gz)`` or ``PROCAR(.gz)`` files)
     - ✓ (PAW projections from the ``.gpw`` files)
   * - Bulk DOS parsing (for Fermi level & concentration analysis)
     - ✓ (``vasprun.xml(.gz)`` files)
     - ✗ (a ``pymatgen`` ``FermiDos`` object can be supplied instead)
   * - Spin-orbit coupling (SOC)
     - ``vasp_ncl`` inputs (by default if any element has an atomic number
       Z >= 31), and parsing
     - No input generation; non-collinear outputs can be parsed
   * - Compatibility checks when parsing
     - ``INCAR``, ``KPOINTS`` and ``POTCAR`` settings
     - Mode (and plane-wave cutoff), functional, `k`-points, PAW datasets,
       basis set and real-space grid

In each case, the Kumagai (eFNV) correction is used by default when the outputs
it requires are available, and otherwise the Freysoldt (FNV) correction.

Default Settings
----------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * -
     - ``VASP``
     - ``GPAW`` (experimental)
   * - Settings files
     - |RelaxSet.yaml|, |DefectSet.yaml|, |HSESet.yaml|,
       |SinglePointSet.yaml|, |ConvergenceSet.yaml|, |PotcarSet.yaml|
     - |GPAW_DefectSet.yaml|
   * - Functional
     - HSE06 hybrid DFT (set ``LHFCALC = False`` in ``user_incar_settings``
       for semi-local DFT), with semi-local (GGA) DFT for the competing phase
       `k`-point convergence tests
     - HSE06 hybrid DFT (``xc``; which requires plane-wave mode)
   * - Basis set
     - Plane waves with ``ENCUT = 300`` eV (350 eV for the competing phase
       `k`-point convergence tests)
     - Plane waves with a 400 eV cutoff (``mode``)
   * - `k`-points
     - Reciprocal density of 100 Å\ :sup:`-3` for defect supercells
       (``vasp_gam`` if this is Γ-point only), and 64 / 200 Å\ :sup:`-3` for
       non-metallic / metallic competing phases
     - Γ-point only
   * - Relaxations
     - Fixed cell (``ISIF = 2``), to forces below 0.01 eV/Å (``EDIFFG``)
     - Fixed cell, with the ``PreconLBFGS`` optimiser, to forces below
       0.01 eV/Å (``fmax``)
   * - Input structures
     - ``POSCAR`` files are only written when requested (``positions=True``,
       or ``vasp_gam=True`` for the ``vasp_gam`` folders), then rattled by
       default for relaxations (not for single-point calculations), as
       structures are expected to be taken from structure-searching (e.g.
       with |ShakeNBreak|) or the preceding calculation stage
     - ``structure.cif`` files are always written, rattled by default for
       defect relaxations
   * - Pseudopotentials
     - PBE ``POTCAR``\s, as chosen in |PotcarSet.yaml|
     - ``GPAW`` default PAW datasets
   * - Folders
     - ``<defect>/<stage>/``, and ``<formula>_bulk/<stage>/`` for the
       reference bulk supercell
     - ``<defect>/``, and ``<formula>_bulk/`` for the reference bulk
       supercell

Further Information
-------------------

- ``VASP``: See the :doc:`Tutorials` and :doc:`Tips` pages, which use ``VASP``
  throughout.
- ``GPAW``: See the :doc:`GPAW_Support` page for installation, usage and the
  current limitations (support is experimental).
