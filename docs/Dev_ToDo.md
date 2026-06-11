# `doped` Development To-Do List
## Defect Complexes
- Generation (see https://github.com/SMTG-Bham/doped/issues/91, SK split vacancies additions and `Future_ToDo.md`)
- Parsing

## Post-processing / analysis / plotting
- Better automatic defect formation energy plot colour handling (set similar colours for similar defects (types and inequivalent sites)) – and more customisable?
  - `aide` labelling of defect species in formation energy plots? See `labellines` package for this (as used in our chempot heatmap plotting)
  - Option for degeneracy-weighted ('reduced') formation energy diagrams, similar to reduced energies in SOD. See Slack discussion and CdTe pyscfermi notebooks. Would be easy to implement if auto degeneracy handling implemented.
  - Could also add an optional right-hand-side y-axis for defect concentration (for a chosen anneal temp) to our TLD plotting (e.g. `concentration_T = None`) as done for thesis, noting in docstring that this obvs doesn't account for degeneracy!
  - Separate `dist_tol` for interstitials vs (inequivalent) vacancies/substitutions? (See Xinwei chat) Any other options on this front?
  - Also see Fig. 6a of the `AiiDA-defects` preprint, want plotting tools like this?
- Charge corrections for polarons; code there, just need to allow inputs of bare calculation outputs (and then can extend to allow polaron input file generation and parsing/plotting). Then update ``ShakeNBreak_Polaron_Workflow`` example with this too.
- Kumagai GKFO and CC diagram corrections. Implemented in `pydefect` and relatively easy to port?

## Docs
- Add our recommended workflow (gam, NKRED, std, ncl). See https://sites.tufts.edu/andrewrosen/density-functional-theory/vasp/ for some possibly useful general tips.
- Workflow diagram with: https://twitter.com/Andrew_S_Rosen/status/1678115044348039168?s=20
- Example on docs (miscellaneous/advanced analysis tutorial page?) for adding entries / combining multiple `DefectThermodynamics` objects
- Readily-usable in conjunction with `atomate`, `AiiDA`(-defects), `vise`, `CarrierCapture`, and give some
  quick examples? Add as optional dependencies.

- Show usage of `get_conv_cell_site` in notebooks/docs (in an advanced analysis tutorial with other possibly useful functions being showcased?)
- The Wyckoff analysis code is very useful. See
  https://github.com/spglib/spglib/issues/135. Should describe and exemplify this in the docs (i.e. the
  `get_wyckoff_label_and_equiv_coord_list()` from just a `pymatgen` site and spacegroup).
- Note that charge states are guessed based on different factors, but these rely on auto-determined
  oxidation states and can fail in weird cases. As always please consider if these charge states are
  reasonable for the defects in your system. (i.e. low-symmetry, amphoteric, mixed-valence cases etc!)
  - Note cases where we expect default charge states to not be appropriate (e.g. mixed ionic-covalent systems, low-symmetry systems and/or with amphoteric species), often better to test more than necessary to be thorough! (And link Xinwei stuff, Ke F_i +1 (also found with our Se and Alex's Ba2BiO6)) – i.e.
    use your head!
  - And particularly when you've calculated your initial set of defect results! E.g. with Sb2Se3, all antisites and interstitials amphoteric, so suggests you should re-check amphotericity for all vacancies
- Note about rare cases where `vasp_gam` pre-relaxation can fail (e.g. Wenzhen's case); extremely disperse bands with small bandgaps, where low k-point sampling can induce a phase transition in the bulk structure. In these cases, using a special k-point is advised for the pre-relaxations. You can get the corresponding k-point for your supercell (given the primitive cell special k-point) using the `get_K_from_k` function from `easyunfold`, with the `doped` `supercell_matrix`.
- Show quick example case of the IPR code from `pymatgen-analysis-defects` (or from Adair code? or others?)
- Note about cation-anion antisites often not being favourable in ionic systems, may be unnecessary to calculate (and you should think about the charge states, can play around with `probability_threshold` etc).
- Should flick through other defect codes (see
  https://shakenbreak.readthedocs.io/en/latest/Code_Compatibility.html, also `AiiDA-defects`) and see if
  there's any useful functionality we want to add!

## SK To-Do for next update:
- Update SnB requirement (and thus doped) to pmg-core>2026.5.23 after ROPT fix; https://github.com/materialsproject/pymatgen-core/pull/69
- Finish ``prune_to_expected_polymorphs`` testing and handling of new ``mp-api`` behaviour with default thermo types (may need to update requirement); https://github.com/materialsproject/api/issues/1104, https://github.com/materialsproject/api/pull/1087 -- drafts in SK shelved changes 
- It will also be good to use the `scan_X` functions now in the main thermodynamics tutorial as this should now be the most convenient and recommended way of doing this, unless extra control is needed e.g. to do the bandgap scissoring shown for CdTe. Keep old code for reference at the bottom maybe? With delta_VBM/CBM as a function example 
  - Ideally, would implement the general ``scan()`` function and then showcase it in the tutorials?
- Tutorials general structure clean-up?
- Update all tutorial notebooks to use latest codebase
- Add example to chemical potentials / thermodynamics analysis tutorials of varying chemical potentials as a function of temperature/pressure (i.e. gas phases), using the `Spinney` functions detailed here (https://spinney.readthedocs.io/en/latest/tutorial/chemipots.html#including-temperature-and-pressure-effects-through-the-gas-phase-chemical-potentials) or possibly `DefAP` functions otherwise. Xinwei Sb2S3 stuff possibly a decent example for this, see our notebooks. TODO in ``doped.chemical_potentials``.
- Re-run pytest timings (with proper heavy-test skipping now, updated pmg `LOCPOT` parsing, many test changes etc)
- For v4.1; search for deprecation warnings, parameter-order warnings and remove (mostly flagged by TODOs) -- also a couple in SnB to remove.

delta_VBM/CBM in notebooks. say it can be important to not assume symmetric shifts, esp with differences in eff masses (though note Ganose & Ling found the symmetric shift was fine??), and link to these papers
