Adding Support for a New Calculator
========================================

All calculator-specific code in ``doped`` lives in the ``doped.io``
subpackage, with each supported calculator having a
``doped.io.<calculator>`` subpackage:

.. code-block::

    doped/io/
    ├── inputs.py         # calculator-agnostic DefectsSetBase input-generation base class
    ├── outputs.py        # calculator-agnostic CalculationOutputs container
    ├── utils.py          # calculator-agnostic helpers (calculation file/folder discovery)
    ├── serialized/
    │   └── outputs.py    # escape-hatch backend for pre-serialised CalculationOutputs JSONs
    └── vasp/
        ├── inputs.py     # input file generation (DefectsSet etc.)
        └── outputs.py    # output parsing (vasprun.xml, OUTCAR, LOCPOT...)

The rest of the ``doped`` codebase (defect generation, structure/site
analysis, charge corrections, parsing, eigenvalue analysis, thermodynamics,
chemical potentials...) is calculator-agnostic, communicating with
calculator-specific parsing code via the
:class:`~doped.io.outputs.CalculationOutputs` container (and, for competing
phases, ``pymatgen`` ``ComputedStructureEntry``\ s) and the backend protocol
described below. Calculation `input` generation is likewise dispatched to the
calculator backend (``doped.io.<calculator>.inputs``), from
:class:`~doped.generation.DefectsGenerator` (defect supercells) and
:class:`~doped.chemical_potentials.CompetingPhases` (competing phases).

Note that several calculator-agnostic functions/methods also accept
calculator-native objects directly (e.g. ``Vasprun``, ``Procar``, ``Locpot``
and ``Outcar`` objects in :class:`~doped.core.DefectEntry` and
``doped.corrections``, or a ``Vasprun`` for the bulk DOS in
``doped.thermodynamics``). These type unions are deliberate
backwards-compatibility conveniences for VASP users, and are always
alternatives to the calculator-agnostic inputs (paths parsed via the
backend, ``CalculationOutputs``, ``FermiDos``...) -- not requirements.

Step 1: Output Parsing
--------------------------

Create ``doped/io/<calculator>/outputs.py`` implementing:

.. code-block:: python

    def get_calculation_outputs(path: PathLike, **kwargs) -> CalculationOutputs: ...

    CALC_OUTPUT_MASK = ("<output filename pattern(s)>",)  # e.g. ("vasprun.xml", "vasprun.xml.gz")

..

   ``CALC_OUTPUT_MASK[0]`` is used as the canonical output filename in file
   discovery and warning messages, so it should be a substring of the actual
   output filenames (as with ``"vasprun.xml"`` matching both
   ``vasprun.xml`` and ``vasprun.xml.gz``).

which parses the outputs of a supercell calculation in ``path`` to a
:class:`~doped.io.outputs.CalculationOutputs` object, and declares the
filename pattern(s) identifying calculation output files (used for
calculation folder discovery). ``get_calculation_outputs()`` should accept
(and may ignore) the generic keyword arguments used by ``doped``'s parsing
machinery: ``label`` (``"bulk"``/``"defect"``, for informative warnings) and
``parse_projected_eigen``. Only ``structure`` (the final relaxed structure)
and ``energy`` (final total energy in eV) are required in
``CalculationOutputs``; the optional attributes unlock specific analyses:

.. list-table::
   :header-rows: 1

   * - ``CalculationOutputs`` attributes
     - Analyses enabled
   * - ``planar_averaged_potentials``
     - Freysoldt (FNV) finite-size charge corrections
   * - ``site_potentials``
     - Kumagai (eFNV) finite-size charge corrections (anisotropic systems)
   * - ``vbm``, ``cbm``, ``band_gap``
     - Formation energy diagrams (bulk reference calculation)
   * - ``eigenvalues``, ``projected_eigenvalues``, ``kpoint_coords``,
       ``kpoint_weights``, ``efermi``
     - Electronic eigenvalue analysis (band-edge & in-gap states, shallow
       defect identification)
   * - ``nelect``, ``charge``, ``magnetization``
     - Automatic charge-state determination and spin degeneracies (for
       concentrations / Fermi-level analysis)
   * - ``run_metadata``
     - Bulk/defect calculation compatibility checks

If your calculator does not report the total electron count directly, but does
give the band eigenvalues & occupancies, then
:func:`~doped.io.outputs.nelect_from_eigenvalues` can be used to determine
``nelect`` from these (and the `k`-point weights).

With this, ``doped.io.get_calculation_outputs(path, calculator="<calculator>")``
dispatches to your parser, and -- more importantly --
:class:`~doped.parsing.DefectsParser`\ /\ :class:`~doped.parsing.DefectParser`
can parse full defect calculation sets with
``DefectsParser(..., calculator="<calculator>")``.

Further optional module-level constants/functions extend the supported
functionality (all implemented by ``doped.io.vasp.outputs``, as the
reference backend):

- ``SUBFOLDER_PRIORITY``: priority-ordered calculation subfolder names for
  auto-detection (e.g. ``["vasp_ncl", ..., "vasp_gam"]`` for VASP).
- ``FILE_PARSING_ACTIONS``: ``{file type: what it is parsed for}``, for
  grouped multiple-files warnings.
- ``get_planar_averaged_potentials(path, dir_type, quiet)`` /
  ``get_site_potentials(path, dir_type, quiet, outputs, total_energy)`` (and
  the corresponding ``PLANAR_POTENTIALS_FILE`` / ``SITE_POTENTIALS_FILE``
  file-name constants): lazy loading of charge-correction data, when not
  parsed up-front into ``CalculationOutputs``.
- ``check_run_compatibility(defect_outputs, bulk_outputs, warn)`` (and
  ``MISMATCH_WARNING_SPECS``): defect/bulk calculation settings
  compatibility checks, populating ``"run_metadata"`` (and any
  ``"mismatching_..."`` entries) in ``DefectEntry.calculation_metadata``.
- ``check_entry_compatibility(entries, template_candidates)``: the same
  check for competing phase entries, comparing each against a reference
  (template) entry and warning about mismatched calculation settings, for
  :class:`~doped.chemical_potentials.CompetingPhasesAnalyzer` (with
  ``check_compatibility=True``, the default). Entries lacking the relevant
  settings data are skipped.
- ``load_eigenvalue_outputs(path, outputs, projections, label, run_metadata)``:
  loading of outputs `with orbital projections` for eigenvalue / band-edge
  analysis (``DefectEntry.get_eigenvalue_analysis()``), when not parsed
  up-front. ``outputs``/``projections`` are optional already-loaded
  calculator-native objects or file paths (e.g. ``Vasprun``/``Procar``
  objects with VASP).
- ``get_fermi_dos(path) -> (FermiDos, vbm, band_gap)``: parsing of a bulk
  DOS calculation to a ``pymatgen`` ``FermiDos`` (plus the VBM eigenvalue &
  band gap from that calculation, for consistency checking against the bulk
  supercell), for Fermi level / carrier concentration analysis with
  :class:`~doped.thermodynamics.DefectThermodynamics`. Without this,
  ``FermiDos`` objects can still be constructed directly by the user and
  passed as ``bulk_dos``.
- ``get_competing_phase_entry(path) -> ComputedStructureEntry``: parsing of
  a competing phase (bulk crystal) calculation to a ``pymatgen``
  ``ComputedStructureEntry``, for chemical potential analysis with
  :class:`~doped.chemical_potentials.CompetingPhasesAnalyzer`
  (``CompetingPhasesAnalyzer(..., calculator="<calculator>")``). Calculation
  summary info & settings should be added to ``entry.data``, including
  ``"converged_electronic"``/``"converged_ionic"`` (convergence warnings) and
  ``"folder"`` (the parsed calculation folder). Without this,
  ``CompetingPhasesAnalyzer`` can still be initialised directly from a list
  of ``ComputedEntry``/``ComputedStructureEntry`` objects.

The generic calculation-file discovery helpers in ``doped.io.utils``
(e.g. ``_find_calc_outputs``, ``_determine_subfolder``) are parameterised by
calculator-specific filename masks and subfolder priority lists (see their
VASP-defaulted wrappers in ``doped.io.vasp.outputs``), and can be reused
when implementing output parsing for other calculators.

Alternatively, for one-off usage with an unsupported calculator, the
``doped.io.serialized`` escape-hatch backend can be used without writing any
backend code: construct the ``CalculationOutputs`` objects yourself (however
you like), save each to its calculation directory with
``dumpfn(outputs, "<dir>/calculation_outputs.json.gz")``, and parse with
``DefectsParser(..., calculator="serialized")``.

Step 2 (optional): Input File Generation
--------------------------------------------

Create ``doped/io/<calculator>/inputs.py`` with input-set classes mirroring
``doped.io.vasp.inputs`` (e.g. ``DefectsSet``, with ``write_files()``
methods), using the defect supercells from
:class:`~doped.generation.DefectsGenerator`. The calculator-agnostic
orchestration (input formatting/naming, per-defect input-set dictionaries,
folder-structure writing & provenance serialisation) is provided by
:class:`~doped.io.inputs.DefectsSetBase` -- subclass it and implement the
``_defect_input_set()`` & ``_write_defect()`` hooks (see the VASP
``DefectsSet`` for reference). Per-calculation input sets should be built on
the ``pymatgen.io.core`` ``InputSet``/``InputGenerator`` framework where
available for the calculator (e.g. ``VaspInputSet``, on which
``doped.io.vasp.inputs.DopedDictSet`` is based). Default calculation
parameters should live in data files alongside the module (as with
``doped/io/vasp/VASP_sets``), so users can inspect and override them.

Competing phase calculation inputs (for chemical potential limits) are
generated by the same module, via these functions:

.. code-block:: python

    def get_kpoint_convergence_sets(competing_phases, **kwargs) -> dict[str, Any]: ...
    def get_relaxation_sets(competing_phases, **kwargs) -> dict[str, Any]: ...
    def get_singlepoint_sets(competing_phases, **kwargs) -> dict[str, Any]: ...
    def write_input_sets(input_sets, **kwargs) -> dict[str, Any]: ...
    def write_kpoint_convergence_files(competing_phases, **kwargs) -> dict[str, Any]: ...
    def write_relaxation_files(competing_phases, **kwargs) -> dict[str, Any]: ...
    def write_singlepoint_files(competing_phases, **kwargs) -> dict[str, Any]: ...

each returning a ``{output folder path: input set}`` dictionary. The
corresponding :class:`~doped.chemical_potentials.CompetingPhases` methods are
thin dispatchers to these (``CompetingPhases(..., calculator="<calculator>")``),
forwarding all keyword arguments -- so the calculator-specific options (e.g.
``user_incar_settings`` / ``user_potcar_settings`` with ``VASP``) and their
documentation live in the backend. ``CompetingPhases`` provides the
calculator-agnostic parts: which phases to calculate, their categorisation
(``"metals"`` / ``"non-metals"`` / ``"molecules"``, which set the smearing
and `k`-point sampling choices), structures, and output folder names, all
yielded by ``CompetingPhases._iter_entries_with_categories()``.

Step 3: Tests
-----------------

Add tests using example calculation data under
``tests/data/<calculator>/``, mirroring the structure of
``tests/data/vasp/``. At minimum, test that ``get_calculation_outputs()``
correctly populates the container for a bulk and a defect supercell
calculation (including a charged defect, for the potential-based correction
data). The ``doped.io.serialized`` backend tests in ``tests/test_io.py``
(checking full ``DefectsParser`` equivalence between direct VASP output
parsing and the calculator-agnostic pathway) provide a useful template.
