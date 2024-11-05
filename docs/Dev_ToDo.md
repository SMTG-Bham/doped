# `doped` Development To-Do List
## Chemical potential
- Need to recheck validity of approximations used for extrinsic competing phases (and code for this). Proper `vasp_std` setup (with `NKRED` folders like for defect calcs) and `vasp_ncl` generation.
- Efficient generation of competing phases for which there are many polymorphs? See SK notes from CdTe competing phases.

## Defect Complexes
- Generation (see https://github.com/SMTG-Bham/doped/issues/91 and `Future_ToDo.md`)
- Parsing

## Post-processing / analysis / plotting
- Better automatic defect formation energy plot colour handling (auto-change colormap based on number of defects, set similar colours for similar defects (types and inequivalent sites)) – and more customisable?
  - `aide` labelling of defect species in formation energy plots? See `labellines` package for this (as used in `pymatgen-analysis-defects` chempots plotting)
  - Ordering of defects plotted (and thus in the legend) should be physically relevant (whether by energy, or defect type etc.)
  - Should have `ncols` as an optional parameter for the function, and auto-set this to 2 if the legend height exceeds that of the plot
  - Don't show transition levels outside of the bandgap (or within a certain range of the band edge, possibly using `pydefect` delocalisation analysis?), as these are shallow and not calculable with the standard supercell approach.
  - Option for degeneracy-weighted ('reduced') formation energy diagrams, similar to reduced energies in SOD. See Slack discussion and CdTe pyscfermi notebooks. Would be easy to implement if auto degeneracy handling implemented.
  - Could also add an optional right-hand-side y-axis for defect concentration (for a chosen anneal temp) to our TLD plotting (e.g. `concentration_T = None`) as done for thesis, noting in docstring that this obvs doesn't account for degeneracy!  
  - Separate `dist_tol` for interstitials vs (inequivalent) vacancies/substitutions? (See Xinwei chat) Any other options on this front?
  - Also see Fig. 6a of the `AiiDA-defects` preprint, want plotting tools like this
- Kumagai GKFO and CC diagram corrections. Implemented in `pydefect` and relatively easy to port?
- 2D corrections; like `pydefect_2d`, or recommended to use SCPC in VASP?
- Can we add an option to give the `pydefect` defect-structure-info output (shown here https://kumagai-group.github.io/pydefect/tutorial.html#check-defect-structures) – seems quite useful tbf

## Housekeeping
- Tutorials general structure clean-up
- Remnant TODOs in code

- Docs:
  - Update note at end of thermo tutorial to link to py-sc-fermi/doped interface.
  - Barebones tutorial workflow, as suggested by Alex G.
  - Add note about `NUPDOWN` for triplet states (bipolarons or dimers (e.g. C-C in Si apparently has ~0.5 eV energy splitting (10.1038/s41467-023-36090-2), and 0.4 eV for O-O in STO from Kanta, but smaller for VCd bipolaron in CdTe))).
  - Add our recommended  workflow (gam, NKRED, std, ncl). See https://sites.tufts.edu/andrewrosen/density-functional-theory/vasp/ for some possibly useful general tips.
  - Show on chemical potentials docs how chempots can be later set as attribute for DefectThermodynamics (loaded from `json`) (e.g. if user had finished and parsed defect calculations first, and then finished chemical potential calculations after).
  - Example on docs (miscellaneous/advanced analysis tutorial page?) for adding entries / combining multiple DefectThermodynamics objects
  - Note that bandfilling corrections are no longer supported, as in most cases they shouldn't be used anyway, and if you have band occupation in your supercell then the energies aren't accurate anyway as it's a resonant/shallow defect, and this is just lowering the energy so it sits near the band edge (leads to false charge state behaviour being a bit more common etc). If the user wants to add bandfilling corrections, they can still doing this by calculating it themselves and adding to the `corrections` attribute. (Link our code in old `pymatgen` for doing this)
  - Regarding competing phases with many low-energy polymorphs from the Materials Project; will build
    in a warning when many entries for the same composition, say which have database IDs, warn the user
    and direct to relevant section on the docs -> Give some general foolproof advice for how best to deal
    with these cases (i.e. check the ICSD and online for which is actually the groundstate structure,
    and/or if it's known from other work for your chosen functional etc.)
  - Show our workflow for calculating interstitials (see docs Tips page, i.e. `vasp_gam` relaxations first (can point to defects tutorial for this)) -> Need to mention this in the defects tutorial, and point to discussion in Tips docs page.
  - Add mini-example of calculating the dielectric constant (plus convergence testing with `vaspup2.0`) to docs/examples, and link this when `dielectric` used in parsing examples. Should also note that the dielectric should be in the same xyz Cartesian basis as the supercell calculations (likely but not necessarily the same as the raw output of a VASP dielectric calculation if an oddly-defined primitive cell is used)
  - `vasp_ncl` chemical potential calculations for metals, use `ISMEAR = -5`, possibly `NKRED` etc. (make a function to generate `vasp_ncl` calculation files with `ISMEAR = -5`, with option to set different kpoints) - if `ISMEAR = 0` - converged kpoints still prohibitively large, use vasp_converge_files again to check for quicker convergence with ISMEAR = -5.
  - Often can't use `NKRED` with `vasp_std`, because we don't know beforehand the kpts in the IBZ (because symmetry on for `vasp_std` chempot calcs)(same goes for `EVENONLY = True`).
  - Worth noting that for metals it may sometimes be preferable to use a larger cell with reduced kpoints, due to memory limitations.
  - Readily-usable in conjunction with `atomate`, `AiiDA`(-defects), `vise`, `CarrierCapture`, and give some
    quick examples? Add as optional dependencies.
  - Workflow diagram with: https://twitter.com/Andrew_S_Rosen/status/1678115044348039168?s=20
  - Setting `LREAL = Auto` can sometimes be worth doing if you have a very large supercell for speed up, _but_ it's important to do a final calculation with `LREAL = False` for accurate energies/forces, so only do if you're a power user and have a very large supercell.
  - Show usage of `get_conv_cell_site` in notebooks/docs (in an advanced analysis tutorial with other possibly useful functions being showcased?)
  - Note in docs that `spglib` convention used for Wyckoff labels and conventional structure definition.
    Primitive structure can change, as can supercell / supercell matrix (depending on input structure,
    `generate_supercell` etc), but conventional cell should always be the same (`spglib` convention).
  - Add examples of extending analysis with `easyunfold` and `py-sc-fermi`, and get the lads to add
    this to their docs as example use cases as well. Also include examples of extending to
    non-radiative carrier capture calcs with `CarrierCapture.jl` and `nonrad`. Show example of using
    `sumo` to get the DOS plot of a defect calc, and why this is useful.
  - Note about SOC for chemical potential calculations ([FERE paper](https://doi.org/10.1103/PhysRevB.
    85.115104) suggests that the SOC effects on total energy cancel out for chemical potential
    calculations, but only the case when the occupation of the SOC-affected orbitals is constant
    (typically not the case)) Better to do consistently (link Emily SOC work and/or thesis).
    - But, can generally use non-SOC energies to reliably determine relative energies of polymorphs of the same composition (oxidation states), to good accuracy.
    - Also, can use symmetry with SOC total energy calculations, have tested this.
  - Link to Irea review, saying that while spin and configurational degeneracies are accounted for
    automatically in `doped`, excited-state degeneracy (e.g. with bipolarons/dimers with single and triplet
    states) are not, so the user should manually account for this if present. Also note that
    temperature effects can be important in certain cases so see this review if that's the case.
  - Should have recommendation somewhere about open science practices. The doped defect dict and thermo jsons should always be shared in e.g. Zenodo when publishing, as contains all info on the parsed defect data in a lean format. Also using the `get_formation_energies` etc. functions for SI tables is recommended.
  - Add our general rule-of-thumbs/expectations regarding charge corrections:
    - Potential alignment terms should rarely ever be massive
    - In general, the correction terms should follow somewhat consistent trends (for a given charge state, across defects), so if you see a large outlier in the corrections, it's implying something odd is happening there. This is can be fairly easily scanned with `get_formation_energies`.
  - The Wyckoff analysis code is very useful and no other package can do this afaik. See
    https://github.com/spglib/spglib/issues/135. Should describe and exemplify this in the docs (i.e. the
    `get_wyckoff_label_and_equiv_coord_list()` from just a `pymatgen` site and spacegroup 🔥) and JOSS
    paper.
  - Note that charge states are guessed based on different factors, but these rely on auto-determined
    oxidation states and can fail in weird cases. As always please consider if these charge states are
    reasonable for the defects in your system. (i.e. low-symmetry, amphoteric, mixed-valence cases etc!)
    - Note cases where we expect default charge states to not be appropriate (e.g. mixed ionic-covalent systems, low-symmetry systems and/or with amphoteric species), often better to test more than necessary to be thorough! (And link Xinwei stuff, Ke F_i +1 (also found with our Se and Alex's Ba2BiO6)) – i.e.
      use your f*cking head!
    - And particularly when you've calculated your initial set of defect results! E.g. with Sb2Se3, all antisites and interstitials amphoteric, so suggests you should re-check amphotericity for all vacancies
  - Note about rare cases where `vasp_gam` pre-relaxation can fail (e.g. Wenzhen's case); extremely disperse bands with small bandgaps, where low k-point sampling can induce a phase transition in the bulk structure. In these cases, using a special k-point is advised for the pre-relaxations. You can get the corresponding k-point for your supercell (given the primitive cell special k-point) using the `get_K_from_k` function from `easyunfold`, with the `doped` `supercell_matrix`.
  - Show quick example case of the IPR code from `pymatgen-analysis-defects` (or from Adair code? or others?)
  - Note (in tutorial when showing thermodynamic analysis?) that the spin degeneracy is automatically guessed based on the number of electrons (singlet if even or doublet if odd), but can be more complicated if you have higher multiplets (e.g. with quantum/magnetic defects, d-orbital defects etc), in which case you can manually adjust (show example). Also note that the VASP output magnetisation should match this, and e.g. a magnetisation of 0 for an odd-electron system is unphysical! – Can mention this at the false charge states discussion part as it's a consequence of this. (Unfortunately can't auto check this in `doped` as is as it's only in `pymatgen` `Outcar` objects not `Vasprun`)
  - Brief mention in tips/advanced docs about possibly wanting to do dielectric/kpoints-sampling weighted supercell generation (if user knows what they're doing, they can use a modified version of the supercell generation algorithm to do this), and in general if kpoints can be reduced with just a slightly larger supercell, then this is often preferable.
  - Note about cation-anion antisites often not being favourable in ionic systems, may be unnecessary to calculate (and you should think about the charge states, can play around with `probability_threshold` etc).
- Should flick through other defect codes (see
  https://shakenbreak.readthedocs.io/en/latest/Code_Compatibility.html, also `AiiDA-defects`) and see if
  there's any useful functionality we want to add!
- Add short example notebook showing how to generate a defect PES for carrier capture or NEB calculations (don't use `IBRION = 2` for NEB).

## SK To-Do for next update:
- `doped` repo/docs cleanup `TODO`s above
- Quick run through tutorial notebooks to check code all updated and running.
- Clean up repo, removing old unnecessary git blobs
- Note in chempots tutorial that LaTeX table generator website can also be used with the `to_csv()` function to generate LaTeX tables for the competing phases.
- Add note to chempots tutorial that if your bulk phase is lower energy than its version on the MP (e.g. distorted perovskite), then you should use this for your bulk competing phase calculation.
- - Should have a general refactor from `(bulk, defect)` to `(defect, bulk)` in inputs to functions (e.g. site-matching, symmetry functions etc), as this is most intuitive and then keep consistent throughout.
- Quick-start tutorial suggested by Alex G
- Test chempot grid plotting tool.
- `dist_tol` should also group defects for the concentration etc functions, currently doesn't (e.g. `CdTe_thermo.get_equilibrium_concentrations(limit="Te-rich", per_charge=False, fermi_level=0.5)` and `CdTe_thermo.dist_tol=10; CdTe_thermo.get_equilibrium_concentrations(limit="Te-rich", per_charge=False, fermi_level=0.5)`, same output)
- Add example to chemical potentials / thermodynamics analysis tutorials of varying chemical potentials as a function of temperature/pressure (i.e. gas phases), using the `Spinney` functions detailed here (https://spinney.readthedocs.io/en/latest/tutorial/chemipots.html#including-temperature-and-pressure-effects-through-the-gas-phase-chemical-potentials) or possibly `DefAP` functions otherwise.
- Deal with cases where "X-rich"/"X-poor" corresponds to more than one limit (pick one and warn user?)(e.g. Wenzhen Si2Sb2Te6). Can see `get_chempots` in `pmg-analysis-defects` for inspo on this.