# `doped` Future Development WishList
## Defect calculations set up
- CLI Functionality for core functions.
  - Could also use some of the `snb` functions to add some convenience commands which `cp CONTCAR
    POSCAR` for unconverged `vasp_gam`/`vasp_nkred_std`/`vasp_std` calculations, and copies `CONTCAR`s
    to next VASP folder when converged and submits job.
- Defect complexes: Functionality to setup and parse calculations – can do this with new `pymatgen`
  code? Note that our defect-centring code is currently not implemented for this!
- Add input file generation for FHI-AIMs, CP2K, Quantum Espresso and CASTEP (using SnB functions),
  point to post-processing tools for these online (in docs/example notebooks, `aiida-defects` for  QE,
  https://github.com/skw32/DefectCorrectionsNotebook for AIMs...),
  and give example(s) of how to manually generate a `DefectPhaseDiagram` and chempots from the parsed
  energies of these calculations, so the `doped` analysis tools can then be used.
- Add defect expansion code functionality to regenerate defect structures from a smaller supercell in a
  larger one. Useful for supercell size convergence tests, and accelerating `ShakeNBreak` etc. If/when
  adding, make sure to link in `SnB` docs as well.
  - Related point, using our `doped` site-matching functions, could write some quick functions to plot
    the exponential tailing off of strain / site displacements as we move away from the defect site.
    Could be useful as a validation / check of supercell size convergence, and for quantifying the
    strain / distortion introduced by a certain defect (though I guess the `SnB` tools already do a
    good job of that) – could possibly give a good rule-of-thumb to aim for with a sufficiently large cell?
- Just something to keep in mind; new defect generation code can apparently use oxidation states from
  `defect.defect_structure` and map to defect supercell. Not in our current subclass implementation of
  `Defect`. Is this useful info?
- Ideally, one should be able to input just defect objects somewhere -> an alternative input to `DefectsGenerator`?
Depends where we want supercell generation to happen. Can input to both `DefectsGenerator` or `DefectsSet` (but it'll just send it to `DefectsGenerator` with `kwargs`).

## Post-Processing
- Parsing capability for (non-defect) polarons, so they can then be plotted alongside defects on
  formation energy diagrams. Main things for this are:
  - Input file generation
  - Parsing to determine polaron site (so we can then use charge corrections). Use the site of max
    displacement / bond length difference for this, and future work could be parsing of charge densities
    to get the maximum position. (Note in docs that the user can do this if they want it).
  - General plotting (in transition level diagrams) and analysis (e.g. our site displacement/strain
    functions).
