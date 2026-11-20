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
   :widths: 19 27 27 27

   * -
     - ``VASP``
     - ``GPAW`` (experimental)
     - ``Quantum ESPRESSO`` (experimental)
   * - ``calculator``
     - ``"vasp"`` (default)
     - ``"gpaw"``
     - ``"espresso"``
   * - Defect supercell inputs
     - ✓ (:class:`doped.io.vasp.inputs.DefectsSet`)
     - ✓ (:class:`doped.io.gpaw.inputs.DefectsSet`)
     - ✓ (:class:`doped.io.espresso.inputs.DefectsSet`)
   * - Defect calculation stages
     - ``vasp_gam`` (Γ-point only), ``vasp_nkred_std`` (hybrid DFT with
       ``NKRED``) and ``vasp_std`` relaxations, and ``vasp_ncl`` single-point
       calculations with spin-orbit coupling (SOC), each in its own subfolder
     - One relaxation per defect
     - ``espresso_gamma`` (Γ-point only; optional) and ``espresso_std``
       relaxations, and ``espresso_ncl`` single-point calculations with SOC,
       each in its own subfolder
   * - Competing phase inputs
     - ✓ (`k`-point convergence tests, relaxations and single-point
       calculations)
     - ✗
     - ✓ (`k`-point convergence tests, relaxations and single-point
       calculations)
   * - Defect & competing phase parsing
     - ✓ (|DefectsParser| & |CompetingPhasesAnalyzer|, from
       ``vasprun.xml(.gz)`` files)
     - ✓ (|DefectsParser| & |CompetingPhasesAnalyzer|, from ``.gpw`` restart
       files)
     - ✓ (|DefectsParser| & |CompetingPhasesAnalyzer|, from ``pw.x``
       ``.xml(.gz)`` output files)
   * - Finite-size charge corrections
     - Kumagai (eFNV; from ``OUTCAR(.gz)`` files) and Freysoldt (FNV; from
       ``LOCPOT(.gz)`` files)
     - Kumagai (eFNV) and Freysoldt (FNV), both from the ``.gpw`` files
     - Kumagai (eFNV) and Freysoldt (FNV), both from ``pp.x`` electrostatic
       potential ``.cube`` files
   * - Eigenvalue & orbital projection analysis
     - ✓ (projections from ``vasprun.xml(.gz)`` or ``PROCAR(.gz)`` files)
     - ✓ (PAW projections from the ``.gpw`` files)
     - Implemented (with ``projwfc.x`` projections), but not yet tested
   * - Bulk DOS parsing (for Fermi level & concentration analysis)
     - ✓ (``vasprun.xml(.gz)`` files)
     - ✗ (a ``pymatgen`` ``FermiDos`` object can be supplied instead)
     - Implemented (with ``dos.x`` outputs), but not yet tested
   * - Spin-orbit coupling (SOC)
     - ``vasp_ncl`` inputs (by default if any element has an atomic number
       Z >= 31), and parsing
     - No input generation; non-collinear outputs can be parsed
     - ``espresso_ncl`` inputs (by default if any element has an atomic
       number Z >= 31); parsing of non-collinear outputs is not yet tested
   * - Compatibility checks when parsing
     - ``INCAR``, ``KPOINTS`` and ``POTCAR`` settings
     - Mode (and plane-wave cutoff), functional, `k`-points, PAW datasets,
       basis set and real-space grid
     - ``pw.x`` input parameters, `k`-points and pseudopotentials

In each case, the Kumagai (eFNV) correction is used by default when the outputs
it requires are available, and otherwise the Freysoldt (FNV) correction.

Default Settings
----------------

.. list-table::
   :header-rows: 1
   :widths: 19 27 27 27

   * -
     - ``VASP``
     - ``GPAW`` (experimental)
     - ``Quantum ESPRESSO`` (experimental)
   * - Settings files
     - |RelaxSet.yaml|, |DefectSet.yaml|, |HSESet.yaml|,
       |SinglePointSet.yaml|, |ConvergenceSet.yaml|, |PotcarSet.yaml|
     - |GPAW_DefectSet.yaml|
     - ``QE_Convergence_set.yaml``, ``HSE_set.yaml`` and
       ``QE_PSEUDO_LIBRARY.json`` (in ``doped/io/espresso/QE_sets``)
   * - Functional
     - HSE06 hybrid DFT (set ``LHFCALC = False`` in ``user_incar_settings``
       for semi-local DFT), with semi-local (GGA) DFT for the competing phase
       `k`-point convergence tests
     - HSE06 hybrid DFT (``xc``; which requires plane-wave mode)
     - **Semi-local DFT** (the functional of the pseudopotentials); set
       ``use_hse=True`` for HSE06 hybrid DFT (see below)
   * - Basis set
     - Plane waves with ``ENCUT = 300`` eV (350 eV for the competing phase
       `k`-point convergence tests)
     - Plane waves with a 400 eV cutoff (``mode``)
     - Plane waves with ``ecutwfc = 60`` Ry and ``ecutrho = 240`` Ry
       (placeholders, to be set for the pseudopotentials used)
   * - `k`-points
     - Reciprocal density of 100 Å\ :sup:`-3` for defect supercells
       (``vasp_gam`` if this is Γ-point only), and 64 / 200 Å\ :sup:`-3` for
       non-metallic / metallic competing phases
     - Γ-point only
     - Γ-point only for defect supercells (unless ``kpoints`` or
       ``kpoint_density`` are set), and 64 / 200 Å\ :sup:`-3` for
       non-metallic / metallic competing phases
   * - Relaxations
     - Fixed cell (``ISIF = 2``), to forces below 0.01 eV/Å (``EDIFFG``)
     - Fixed cell, with the ``PreconLBFGS`` optimiser, to forces below
       0.01 eV/Å (``fmax``)
     - Fixed cell (``calculation = 'relax'``), to forces below 0.01 eV/Å
       (``forc_conv_thr``)
   * - Input structures
     - ``POSCAR`` files are only written when requested (``positions=True``,
       or ``vasp_gam=True`` for the ``vasp_gam`` folders), then rattled by
       default for relaxations (not for single-point calculations), as
       structures are expected to be taken from structure-searching (e.g.
       with |ShakeNBreak|) or the preceding calculation stage
     - ``structure.cif`` files are always written, rattled by default for
       defect relaxations
     - Written for the relaxations (rattled by default for defects), but
       not for the ``espresso_ncl`` and competing phase single-point
       calculations (unless ``positions=True``), to which the relaxed
       structures should be added
   * - Pseudopotentials
     - PBE ``POTCAR``\s, as chosen in |PotcarSet.yaml|
     - ``GPAW`` default PAW datasets
     - The filenames in ``QE_PSEUDO_LIBRARY.json`` (no pseudopotentials are
       included with ``doped``; set with ``pseudo_dir`` and ``pseudo_map``)
   * - Folders
     - ``<defect>/<stage>/``, and ``<formula>_bulk/<stage>/`` for the
       reference bulk supercell
     - ``<defect>/``, and ``<formula>_bulk/`` for the reference bulk
       supercell
     - ``<defect>/<stage>/``, and ``<formula>_bulk/<stage>/`` for the
       reference bulk supercell

Further Information
-------------------

- ``VASP``: See the :doc:`Tutorials` and :doc:`Tips` pages, which use ``VASP``
  throughout.
- ``GPAW``: See the :doc:`GPAW_Support` page for installation, usage and the
  current limitations (support is experimental).
- ``Quantum ESPRESSO``: The default functional differs from that with ``VASP``
  and ``GPAW`` (semi-local, rather than hybrid, DFT), as ``pw.x`` only
  supports relaxations and non-collinear (SOC) calculations with hybrid
  functionals for norm-conserving pseudopotentials, while most of the default
  pseudopotential filenames here are for ultrasoft or PAW pseudopotentials.
  Hybrid DFT is still recommended for accurate defect calculations; with
  norm-conserving pseudopotentials (set with ``pseudo_map``), ``use_hse=True``
  uses HSE06 for all defect calculation stages and for the competing phase
  relaxations and single-point calculations. The ``espresso_ncl`` (SOC) and
  competing phase single-point inputs are written without the structure (the
  ``CELL_PARAMETERS`` and ``ATOMIC_POSITIONS`` cards) by default, which
  should be added from the preceding relaxation.
