# `doped` Development To-Do List
## Chemical potential
- Need to recheck validity of approximations used for extrinsic competing phases (and code for this)(see `full_sub_approach_understanding` folder). Proper `vasp_std` setup (with `NKRED` folders like for defect calcs) and `vasp_ncl` generation.
- Efficient generation of competing phases for which there are many polymorphs? See SK notes from CdTe
  competing phases, and notes below.
- See `Future_ToDo.md`.

## Defect Complexes
- Generation (see https://github.com/SMTG-Bham/doped/issues/91, SK split vacancies additions and `Future_ToDo.md`)
- Parsing

## Post-processing / analysis / plotting
- Better automatic defect formation energy plot colour handling (set similar colours for similar defects (types and inequivalent sites)) – and more customisable?
  - `aide` labelling of defect species in formation energy plots? See `labellines` package for this (as used in `pymatgen-analysis-defects` chempots plotting, and our chempot heatmap plotting)
  - Ordering of defects plotted (and thus in the legend) should be physically relevant (whether by energy, or defect type etc.)
  - Option for degeneracy-weighted ('reduced') formation energy diagrams, similar to reduced energies in SOD. See Slack discussion and CdTe pyscfermi notebooks. Would be easy to implement if auto degeneracy handling implemented.
  - Could also add an optional right-hand-side y-axis for defect concentration (for a chosen anneal temp) to our TLD plotting (e.g. `concentration_T = None`) as done for thesis, noting in docstring that this obvs doesn't account for degeneracy!
  - Separate `dist_tol` for interstitials vs (inequivalent) vacancies/substitutions? (See Xinwei chat) Any other options on this front?
  - Also see Fig. 6a of the `AiiDA-defects` preprint, want plotting tools like this
- Charge corrections for polarons; code there, just need to allow inputs of bare calculation outputs (and then can extend to allow polaron input file generation and parsing/plotting). Then update ``ShakeNBreak_Polaron_Workflow`` example with this too.
- Kumagai GKFO and CC diagram corrections. Implemented in `pydefect` and relatively easy to port?
- 2D corrections?
- Can we add an option to give the `pydefect` defect-structure-info output (shown here https://kumagai-group.github.io/pydefect/tutorial.html#check-defect-structures) – seems quite useful tbf


## FermiSolver
- Per-charge outputs are currently not supported for `FermiSolver`, but are for `DefectThermodynamics`, and so only total defect concentrations are accessible. This is useful information to users in a lot of cases, so would be good to include in future. I think this is possible with `py-sc-fermi`, just needs to use the `decomposed` option with `concentration_dict`?
- Should also allow just specifying an extrinsic element for fixed_defects, to allow the user to specify the known concentration of a dopant / over/under-stoichiometry of a given element (but with unknown relative populations of different possible defects) -- realistically the most commonly desired option (but would require a bit of refactoring from the current `py-sc-fermi` implementation). See in-code `TODO` and notes. The DefectThermodynamics JSONs in the repo for extrinsic-doped Selenium (link) would be a good test case for this.
- In future the `fixed_defects`, `free_defects` and `fix_charge_states` options may be added to the `doped` backend (in theory very simple to add, and `doped` currently far quicker ~>10x)
- Add per-site option like in `DefectThermodynamics`, should be quick to add (can use `per_volume=False` in `py-sc-fermi`).
- Show example of extremum position for a defect/carrier concentration occurring at a non-limiting chemical potential (e.g. CdTe from SK thesis, V_S in Sb2S3 just about (https://pubs.acs.org/doi/10.1021/acsenergylett.4c02722)), as this is the main case where the `optimise` function is particularly powerful.
- It will also be good to use the `scan_X` functions now in the main thermodynamics tutorial as this should now be the most convenient and recommended way of doing this, unless extra control is needed e.g. to do the bandgap scissoring shown for CdTe.


## Housekeeping
- Tutorials general structure clean-up
- Remnant TODOs in code

- Docs:
  - Barebones tutorial workflow, as suggested by Alex G.
  - Add our recommended  workflow (gam, NKRED, std, ncl). See https://sites.tufts.edu/andrewrosen/density-functional-theory/vasp/ for some possibly useful general tips.
  - Workflow diagram with: https://twitter.com/Andrew_S_Rosen/status/1678115044348039168?s=20
  - Show on chemical potentials docs how chempots can be later set as attribute for ``DefectThermodynamics`` (loaded from `json`) (e.g. if user had finished and parsed defect calculations first, and then finished chemical potential calculations after).
  - Example on docs (miscellaneous/advanced analysis tutorial page?) for adding entries / combining multiple ``DefectThermodynamics`` objects
  - Regarding competing phases with many low-energy polymorphs from the Materials Project; will build
    in a warning when many entries for the same composition, say which have database IDs, warn the user
    and direct to relevant section on the docs -> Give some general foolproof advice for how best to deal
    with these cases (i.e. check the ICSD and online for which is actually the groundstate structure,
    and/or if it's known from other work for your chosen functional etc.)
  - `vasp_ncl` chemical potential calculations for metals, use `ISMEAR = -5`, possibly `NKRED` etc. (make a function to generate `vasp_ncl` calculation files with `ISMEAR = -5`, with option to set different kpoints) - if `ISMEAR = 0` - converged kpoints still prohibitively large, use vasp_converge_files again to check for quicker convergence with ISMEAR = -5.
    - Worth noting that for metals it may sometimes be preferable to use a larger cell with reduced kpoints, due to memory limitations.
  - Often can't use `NKRED` with `vasp_std`, because we don't know beforehand the kpts in the IBZ (because symmetry on for `vasp_std` chempot calcs)(same goes for `EVENONLY = True`).
  - Readily-usable in conjunction with `atomate`, `AiiDA`(-defects), `vise`, `CarrierCapture`, and give some
    quick examples? Add as optional dependencies.

  - Show usage of `get_conv_cell_site` in notebooks/docs (in an advanced analysis tutorial with other possibly useful functions being showcased?)
  - Add our general rule-of-thumbs/expectations regarding charge corrections:
    - Potential alignment terms should rarely ever be massive
    - In general, the correction terms should follow somewhat consistent trends (for a given charge state, across defects), so if you see a large outlier in the corrections, it's implying something odd is happening there. This is can be fairly easily scanned with `get_formation_energies`.
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
  - Brief mention in tips/advanced docs about possibly wanting to do dielectric/kpoints-sampling weighted supercell generation (if user knows what they're doing, they can use a modified version of the supercell generation algorithm to do this), and in general if kpoints can be reduced with just a slightly larger supercell, then this is often preferable.
  - Note about cation-anion antisites often not being favourable in ionic systems, may be unnecessary to calculate (and you should think about the charge states, can play around with `probability_threshold` etc).
- Should flick through other defect codes (see
  https://shakenbreak.readthedocs.io/en/latest/Code_Compatibility.html, also `AiiDA-defects`) and see if
  there's any useful functionality we want to add!

## SK To-Do for next update:
- `doped` repo/docs cleanup `TODO`s above, and check through code TODOs
- Should have a general refactor from `(bulk, defect)` to `(defect, bulk)` in inputs to functions (e.g. site-matching, symmetry functions etc), as this is most intuitive and then keep consistent throughout?
- Configuration coordinate diagram generation tutorial, linked in other tutorials and codes (CarrierCapture.jl). For defect PESs for carrier capture or NEB calculations (don't use `IBRION = 2` for NEB), and tests.
  - Tests for configuration coordinate diagram generation code
- Stenciling tutorial and tests.
- Quick-start tutorial suggested by Alex G
- Add example to chemical potentials / thermodynamics analysis tutorials of varying chemical potentials as a function of temperature/pressure (i.e. gas phases), using the `Spinney` functions detailed here (https://spinney.readthedocs.io/en/latest/tutorial/chemipots.html#including-temperature-and-pressure-effects-through-the-gas-phase-chemical-potentials) or possibly `DefAP` functions otherwise. Xinwei Sb2S3 stuff possibly a decent example for this, see our notebooks.
- Deal with cases where "X-rich"/"X-poor" corresponds to more than one limit (pick one and warn user?)(e.g. Wenzhen Si2Sb2Te6). Can see `get_chempots` in `pmg-analysis-defects` for inspo on this.
