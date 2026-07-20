Adding Support for a New Calculator
========================================

All calculator-specific code in ``doped`` lives in the ``doped.io``
subpackage, with each supported calculator having a
``doped.io.<calculator>`` subpackage:

.. code-block::

    doped/io/
    ├── outputs.py        # calculator-agnostic CalculationOutputs container
    └── vasp/
        ├── inputs.py     # input file generation (DefectsSet etc.)
        └── outputs.py    # output parsing (vasprun.xml, OUTCAR, LOCPOT...)

The rest of the ``doped`` codebase (defect generation, structure/site
analysis, charge corrections, thermodynamics...) is calculator-agnostic,
communicating with calculator-specific parsing code via the
:class:`~doped.io.outputs.CalculationOutputs` container.

Step 1: Output Parsing
--------------------------

Create ``doped/io/<calculator>/outputs.py`` implementing:

.. code-block:: python

    def get_calculation_outputs(path: PathLike, **kwargs) -> CalculationOutputs: ...

which parses the outputs of a supercell calculation in ``path`` to a
:class:`~doped.io.outputs.CalculationOutputs` object. Only ``structure``
(the final relaxed structure) and ``energy`` (final total energy in eV) are
required; the optional attributes unlock specific analyses:

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
   * - ``nelect``, ``magnetization``
     - Automatic charge-state determination and spin degeneracies (for
       concentrations / Fermi-level analysis)
   * - ``run_metadata``
     - Bulk/defect calculation compatibility checks

With this, ``doped.io.get_calculation_outputs(path, calculator="<calculator>")``
dispatches to your parser, and the parsed data can be provided to the
analysis functions (e.g. charge corrections via
``DefectEntry.get_freysoldt_correction``/``get_kumagai_correction``).

Step 2 (optional): Input File Generation
--------------------------------------------

Create ``doped/io/<calculator>/inputs.py`` with input-set classes mirroring
``doped.io.vasp.inputs`` (e.g. ``DefectsSet``, with ``write_files()``
methods), using the defect supercells from
:class:`~doped.generation.DefectsGenerator`. Default calculation parameters
should live in data files alongside the module (as with
``doped/io/vasp/VASP_sets``), so users can inspect and override them.

Step 3: Tests
-----------------

Add tests using example calculation data under
``tests/data/<calculator>/``, mirroring the structure of
``tests/data/vasp/``. At minimum, test that ``get_calculation_outputs()``
correctly populates the container for a bulk and a defect supercell
calculation (including a charged defect, for the potential-based correction
data).
