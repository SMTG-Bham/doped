Change Log
==========

v.2.4.2
----------
- Allow cases where the calculated host material is unstable wrt competing phases (above the hull), by downshifting to the hull and warning the user about this.
- General updates to chemical potentials code; more robust (better error catches and messages, API key handling), more informative, cleaner outputs.
- Updates to match recent changes in ``pymatgen`` object types (e.g. https://github.com/SMTG-Bham/doped/issues/68)
- Minor efficiency & robustness updates

v.2.4.1
----------
- Speed up eigenvalue parsing by using the faster ``doped`` site-matching functions rather than ``MakeDefectStructureInfo`` from ``pydefect``
- Minor efficiency & robustness updates.
- Minor docs & tutorials updates
- Minor tests updates

v.2.4.0
----------
- Electronic structure analysis by @adair-nicolson & @kavanase:
    - Adds ``DefectEntry.get_eigenvalue_analysis()`` method to plot single-particle eigenvalues and
      analyse orbital character and localisation; usage and examples shown on the
      `docs Tips <https://doped.readthedocs.io/en/latest/Tips.html#eigenvalue-electronic-structure-analysis>`__
      page and the `advanced analysis tutorial <https://doped.readthedocs.io/en/latest/advanced_analysis_tutorial.html#eigenvalue-electronic-structure-analysis>`__.
    - Projected eigenvalues can be parsed from ``vasprun.xml(.gz)`` files (preferred, as more accurate
      with 4 decimal places; c.f. 3 in ``PROCAR(.gz)``; more convenient and only ~5% slower) or ``PROCAR(.gz)``
      files in calculation directories (both with significantly expedited parsing compared to ``pymatgen`` methods).
      Compatible with spin-polarised, unpolarised and SOC calculations. Comes with update by @kavanase to ``easyunfold``
      ``PROCAR.gz`` parsing.
- More efficient defect calculation parsing
- Add ``get_magnetization_from_vasprun`` and ``get_nelect_from_vasprun`` functions to ``doped.utils.parsing``,
  as these attributes are not available from ``pymatgen.io.vasp.outputs.Vasprun``.
- Improve testing efficiency

v.2.3.3
----------
- General robustness updates:
    - Updated file parsing to avoid hidden files.
    - Sanity check in `DefectsGenerator` if input symmetry is `P1`.
    - Add `NKRED` to `INCAR` mismatch tests.
    - Re-parse config & spin degeneracies in concentration/symmetry functions if data not already present
      (if user is porting `DefectEntry`s from older `doped` versions or manually).
    - Avoid unnecessary `DeprecationWarning`s
- Updated docs and linting

v.2.3.2
----------
- Update to match breaking change in ``pymatgen==2024.3.1`` (released today), handling ``incar_params``.

v.2.3.1
----------
- Refactor (phase diagram) ``facet`` to (chemical potential) ``limit`` in ``doped`` chemical potential
  functions, as this is more intuitive for most users.
- Tests updates.
- Minor efficiency/verbosity/robustness/docs improvements.
- Update default ``KPOINTS`` for convergence/production runs in ``chemical_potentials`` based on testing.
- Add optional projections of site displacements upon given vectors by @ireaml

v.2.3.0
----------
- ``DefectsThermodynamics`` class has been added to replace and greatly expand the functionality of the
  ``DefectPhaseDiagram`` object. See tutorials for functionality and usage (plotting, Fermi level /
  concentration analysis, dopability, transition levels (with/out metastable etc).
- Overhaul supercell generation as discussed, now optimises directly off minimum periodic image distance
  (thanks to efficient optimisation algorithm) with some prudent constraints. Significantly reduces
  supercell sizes required in most cases.
- Overhaul defect grouping as discussed, to use the distance between equivalent defect sites (with this
  controllable via the `dist_tol` parameter).
- Add point symmetry and orientational/spin degeneracy parsing, automatically included in thermodynamics
  analysis (and customisable by user).
- Many efficiency improvements (particularly in defect & input file generation, and symmetry functions).
- Check and warning for large defect displacements far from defect site.
- Site displacement (local strain) plotting by @ireaml 🙌
- Auto determination of X-poor/rich facets.
- More control over site selection for eFNV correction.
- Clean, grouped parsing warnings for ``DefectsParser`` (in case many warnings...)
- ``__repr__`` methods for all `doped` classes for informative outputs.
- Tests and tutorials updates.

v.2.2.0
----------
- Added ``DefectsParser`` class for parsing defect calculations:
    - Uses multiprocessing and shared bulk data to massively speed up parsing of many defect supercell
      calcs at once (e.g. from 17 min to < 3 mins for 54 defects in CdTe).
    - Automatically checks ``INCAR``, ``KPOINTS``, ``POTCAR`` and charge correction compatibility between
      all calculations, and warns the user if any are likely to affect formation energies.
- Make ``csv`` input to ``CompetingPhasesAnalyzer`` more flexible, along with other code and docstrings updates.
- Format point group symbol in formation energy plots.
- Refactor ``elt``/``elt_refs`` to ``el/el_refs`` by @adair-nicolson
- Charge states can now be automatically determined even when ``POTCAR``\ s are not setup by the user.

Updates reflected in the ``doped`` parsing tutorial.

v.2.1.0
----------
- Update finite-size defect corrections implementations:
    - ``pydefect`` used directly for eFNV correction (with optimisation for efficiency). Moreover, the
      fully relaxed defect structure (with defect site determined by doped) is used.
    - FNV correction now uses optimised version of ``pymatgen-analysis-defects`` implementation.
    - Updated corrections plotting (much nicer formats, more informative etc)
    - The actual energy error in the correction is now estimated, and the user is warned if this exceeds
      ``error_tolerance`` (optional parameter, 0.05 eV by default)
    - Bandfilling corrections no longer automatically calculated as (1) almost always not recommended
      and (2) will show an example of calculating these if needed using our code in ``pymatgen`` on the docs
- Efficiency improvements in obtaining defect site info (Wyckoff positions)
- Additional utils and functions for defect generation and manipulation.
- (Many) updated tests.
- Added functionality for robustly determining the point group symmetry of _relaxed_ defects 🔥

v.2.0.5
----------
- Update oxi-state handling to:
    - Use pre-assigned oxi states if present
    - Handle ``pymatgen`` oxi-state guessing failures (non-integer oxi states, inaccurate oxi states with
      max_sites, failures for extremely large systems etc)
- Update default ``probability_threshold`` from 0.01 to 0.0075.
- Account for rare possibility of user being on a non UTF-8 system.
- Italicise "V" for vacancy in plotting.
- SMTG-UCL -> SMTG-Bham
- Tests and formatting updates.

v.2.0.4
----------
- Add supercell re-ordering tests for parsing
- Ensure final _relaxed_ defect site (for interstitials and substitutions) is used for finite-size
  charge corrections
- Consolidate functions and input sets with ``ShakeNBreak``
- Update defect generation tests
- Use more efficient Wyckoff determination code

v.2.0.3
----------
- Sort defect entries in ``DefectPhaseDiagram`` for deterministic behaviour (particularly for plotting).
- Tests updates (archive test plots, update extrinsic generation tests etc).
- Avoid long stacklevel issue which cropped up in ``python3.8`` tests for ``SnB``
- Update PDF figure ``savefig`` settings, and add ``_get_backend`` function.

v.2.0.2
----------
- Refactor ``_check_user_potcars()`` to ``DefectDictSet`` rather than ``DefectRelaxSet``, and add ``write_input
  ()`` method (which runs ``_check_user_potcars()`` first).
- Update defect generation tests
- Add troubleshooting docs page and update tips docs page

v.2.0.1
----------
- Update naming handling in ``DefectPhaseDiagram`` to be more robust/flexible, following failure case
  noted by @utf 🙌
- Ensure package data files are correctly included in the package distribution, again noted by @utf 🙌
- Updates to chemical potentials code.
- Refactoring of site-matching code.
- Tests updates and code cleanup.

v.2.0.0
----------
- Major overhaul to rebase onto the new ``pymatgen`` defects code (``>v2022.7.25``).
- Add documentation (https://doped.readthedocs.io/en/latest)
- Add ``DefectsGenerator`` class with major upgrade in functionality.
- Add ``DefectsSet`` classes in ``vasp.py``

v.1.1.2
----------
- Cap ``numpy`` to ``1.23`` to avoid ``pymatgen`` dependency issues.
- Update example workbook to use recommended ``CubicSupercellTransformation``
- Add/remove some ``TODO``\ s

v1.1.1
----------
- ``doped`` now installable from ``conda-forge``! 🎉
- Major overhaul of primary parsing workflow (in ``defect_entry_from_paths()``):
    - Automatic charge-state determination (throwing warning when user specification doesn't match auto-determined)
    - Automatic charge correction determination and application
    - Improved error handling and more informative warning messages
- Add ``test_defectsmaker.py``, ``test_corrections.py`` and ``test_analysis.py`` -> significantly improve test coverage
- Add ``_convert_dielectric_to_tensor()`` function to be more flexible to user input
- Remove old unsupported/deprecated code.
- Add check and warning if multiple output files (``vasprun.xml``/``OUTCAR``/``LOCPOT``) present in bulk/defect directory.
- Minor bug fixes, formatting, docstrings improvement, the usual
- Add and remove ``TODO``\ s


v1.0.6
----------
- Start keeping a ``CHANGELOG``
- ``README`` updates to give step-by-step instructions on setting up MP API key, ``POTCAR``\ s for ``pymatgen`` and virtual ``conda`` environments for ``doped`` and ``ShakeNBreak``
- Major overhaul of ``vasp_input`` functions setup to be far more streamlined and customisable.
- Major overhaul of ``chemical_potentials`` code; now with improved algorithm for selecting potential competing phases
- Update of example notebooks
- Add tests for parsing calculations, chemical_potentials and vasp_input
- Add GH Actions workflows (for tests, GH releases and pypi packaging)
- Adopt recommended versioning convention based on dates
- General tidy up, docstring padding, formatting and ``TODO`` addition/removal
- Ensure all inputs/outputs are ``JSON``\able, now recommending this for better forward/backward compatibility
- Refactor ``dope_stuff`` to ``plotting`` and ``analysis`` to be more clear and PROfessional, yo
- Refactor from hard-coded defaults / slightly-less-human-readable ``json`` files to ``yaml`` files with default settings.
- Refactor ``defectsmaker`` output, more efficient, cleaner and informative
