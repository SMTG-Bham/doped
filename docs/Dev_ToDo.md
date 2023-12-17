# `doped` Development To-Do List
## Defect calculations set up
- See SK Remarkable notes

## Chemical potential
- Update chemical potential tools to work with new Materials Project API. Currently, supplying an API key for the new Materials Project API returns entries which do not have `e_above_hull` as a property, and so crashes. Ideally would be good to be compatible with both the legacy and new API, which should be fairly straightforward (try importing MPRester from mp_api client except ImportError import from pmg then will need to make a whole separate query/search because `band_gap` and `total_magnetisation` no longer accessible from `get_entries`). See https://docs.materialsproject.org/downloading-data/using-the-api
- Currently inputting multiple extrinsic `sub_species` will assume you are co-doping, and will output competing phases for this (e.g. K and In with BaSnO3 will output KInO2), default should not be to do this, but have an optional argument for co-doping treatment.
- Publication ready chemical potential diagram plotting tool as in Adam Jackson's `plot-cplap-ternary` (3D) and Sungyhun's `cplapy` (4D) (see `doped_chempot_plotting_example.ipynb`; code there, just needs to be implemented in module functions). `ChemicalPotentialGrid` in `py-sc-fermi` interface could be quite useful for this? (Worth moving that part of code out of `interface` subpackage?)
  - Also see `Cs2SnTiI6` notebooks for template code for this.
- Functionality to combine chemical potential limits from considering different extrinsic species, to be able to plot defect formation energies for different dopants on the same diagram.
- Once happy all required functionality is in the new `chemical_potentials.py` code (need more rigorous tests, see original pycdt tests for this and make sure all works with new code), showcase all functionality in the example notebook, remove the old modified-pycdt `_chemical_potentials.py` code.
- Should output `json` of Materials Project `ComputedStructureEntry` used for each competing phase directory, to aid provenance.
- Note in tutorial that LaTeX table generator website can also be used with the `to_csv()` function to generate LaTeX tables for the competing phases.

## Post-processing / analysis / plotting
- Should auto-check the magnetisation output; if it comes to around
  zero for an odd-electron defect, suggests getting spurious shallow defect behaviour!
- Try re-determine defect symmetry and site multiplicity (particularly important for interstitials, as
  relaxation may move them to lower/higher symmetry sites which significantly different multiplicity).
  - Should be doable with current point symmetry tools, especially when both the defect and bulk
    structures are available. The configurational degeneracy should be just the final site degeneracy
    (i.e. Wyckoff number) divided by the initial, or equivalently the initial number of symmetry
    operations divided by the final, so we can just use this to determine the final site degeneracies.
    For interstitials, should be based off just the Wyckoff number of the final relaxed site.
    Should make this a parsed defect property, defined relative to the conventional cell (so they
    actually correspond to Wyckoff numbers, will need some idiotproof checks/notes for users about this),
    and have this automatically plug-and-play with `py-sc-fermi` (can do by setting `spin_degeneracy` and `config_degeneracy` properties, and use this in `py-sc-fermi` `interface` code). Already have the site analysis /
    Wyckoff matching code for this.
  - See `pydefect` and pmg `finder.py` for tools for this.
  - For complex defects, this is future work, and should be done manually (note in docs and give
    warning when parsing).
    - For split-interstitials and split-vacancies however, should be relatively straightforward?
      Firstly check that the standard approach described above doesn't happen to work (can test with
      CdTe `Te_i` structures (split-interstitial dimer, oriented along <110> and twisted in different
      charge states; https://doi.org/10.1039/D2FD00043A)). Could determine the centre-of-mass (CoM),
      remove the two split-interstitial atoms, add a dummy species at CoM, get the symm-ops / point
      symmetry of this supercell structure (i.e. with the defect periodic images), then do the same
      with the original structure and get the difference (-> configurational degeneracy) from this. Not
      sure if we can do this in general? Taking the unrelaxed and relaxed defect structures, and
      getting the difference in symm-ops according to `spglib`?
  - Also add consideration of odd/even number of electrons to account for spin degeneracy (can pull from `vr.parameters["NELECT"]` / magnetisation from OUTCAR/vasprun or atomic numbers (even/odd) plus charge)(can also use this to double check odd is odd and even even as expected, warn user if not)
- Complex defect / defect cluster automatic handling. Means we can natively handle complex defects, and
  also important for e.g. `ShakeNBreak` parsing, as in many cases we're ending up with what are
  effectively defect clusters rather than point defects (e.g. V_Sb^+1 actually Se_Sb^-1 + V_Se^+2 in
  Xinwei's https://arxiv.org/abs/2302.04901), so it would be really nice to have this automatic parsing
  built-in, and can either use in SnB or recommend SnB users to check with this.
  - Questions some of our typical expectations of defect behaviour! Actually defect complexes are a bit
    more common than thought.
  - Kumagai's atom-pairing defect analysis code for identifying 'non-trivial' defects is essentially
    this, could be used here?
  - Could do by using the site displacements, with atoms moving outside their vdW radius being flagged
    as (possibly) defective? And see if their stoichiometric sum matches the expected point defect
    stoichiometry. Expected to match one of these transformation motifs:
    - Substitutions:
      - `A_B` -> `A_C` + `C_B`
      - `A_B` -> `A_i` + `V_B`
      - `A_B` -> `A_i` + `C_B` + `V_C`
      - `A_B` -> `C_i` + `A_B` + `V_C` (same defect but inducing a neighbouring Frenkel pair)
    - Vacancies:
      - `V_B` -> `A_B` + `V_A`
      - `A_B` -> 2`V_A` + `A_i` (split-vacancy)
      - `V_B` -> `A_i` + `V_B` + `V_A` (same defect but inducing a neighbouring Frenkel pair)
    - Interstitials:
      - `A_i` -> `A_B` + `B_i`
      - `A_i` -> 2`A_i` + `V_A` (split-interstitial)
      - `A_i` -> `B_i` + `A_i` + `V_B` (same defect but inducing a neighbouring Frenkel pair)
  - How does this change the thermodynamics (i.e. entropic cost to clustering)?
  - In these cases, will also want to be able to plot these in a smart manner on the defect TLD.
    Separate lines to the stoichiometrically-equivalent (unperturbed) point defect, but with the same
    colour just different linestyles? (or something similar)
- Automate `pydefect` shallow defect analysis? At least have notebook showing how to manually do this (Adair's done before?).
  - Should tag parsed defects with `is_shallow` (or similar), and then omit these from plotting/analysis
    (and note this behaviour in examples/docs)
- Better automatic defect formation energy plot colour handling (auto-change colormap based on number of defects, set similar colours for similar defects (types and inequivalent sites)) – and more customisable?
  - `aide` labelling of defect species in formation energy plots? See `labellines` package for this (as used in `pymatgen-analysis-defects` chempots plotting)
  - Ordering of defects plotted (and thus in the legend) should be physically relevant (whether by energy, or defect type etc.)
  - Should have `ncols` as an optional parameter for the function, and auto-set this to 2 if the legend height exceeds that of the plot
  - Don't show transition levels outside of the bandgap (or within a certain range of the band edge, possibly using `pydefect` delocalisation analysis?), as these are shallow and not calculable with the standard supercell approach.
  - Use the update defect name info in `plotting` plotting? i.e. Legend with the inequivalent site naming used in the subscripts?
- Add LDOS plotting, big selling point for defects and disorder!
- Add short example notebook showing how to generate a defect PES / NEB and then parse with fully-consistent charge corrections after (link recent Kumagai paper on this: https://arxiv.org/abs/2304.01454). SK has the code for this in local example notebooks ready to go.
- `transition_levels_table()`. Also ensure we have functionality to print all single-electron TLs (useful to know when deciding what TLs to do carrier capture for. @SeánK has code for this in jupyter notebooks)
- **Optical transitions:** Functions for generating input files, parsing (with GKFO correction) and
  plotting the results (i.e. configuration coordinate diagrams) of optical calculations. Needs to be at
  this point because we need relaxed structures. Sensible naming scheme. Would be useful as this is a
  workflow which ppl often mess up. Can use modified code from `config-coord-plots` (but actually to
  scale and automatically/sensibly parsed etc.)(also see `CarrierCapture` functionalities)
- Option for degeneracy-weighted ('reduced') formation energy diagrams, similar to reduced energies in SOD. See Slack discussion and CdTe pyscfermi notebooks.
- `pydefect` integration? So we can use:
  - Handling of shallow defects
  - Readily automated with `vise` if one wants (easy high-throughput and can setup primitive calcs (BS, DOS, dielectric).
  - Some nice defect structure and eigenvalue analysis
  - GKFO correction
- Showcase `py-sc-fermi` plotting (e.g. from thesis notebook) using `interface` functionality. When doing, add CdTe data as test case for this part of the code. Could also add an optional right-hand-side y-axis for defect concentration (for a chosen anneal temp) to our TLD plotting (e.g. `concentration_T = None`) as done for thesis, noting in docstring that this obvs doesn't account for degeneracy! Also carrier concentration vs Fermi level plots as done in the Kumagai PRX paper? (once properly integrated, add and ask Alex to check/test?)
  Brouwer diagrams; show examples of these in docs using `py-sc-fermi` interface tools. Also see Fig. 6a of the `AiiDA-defects` preprint, want plotting tools like this (some could be PR'd to `py-sc-fermi`)

## Housekeeping
- Clean `README` with bullet-point summary of key features, and sidebar like `SnB`. Add correction plots and other example outputs (see MRS poster for this).
- Code tidy up:
  - Test coverage?
  - Auto-docs generation with little to no warnings/errors?
  - Add type hints for all functions.

- Docs:
  - Create GGA practice workflow, for people to learn how to work with doped and defect calculations
  - Add note about `NUPDOWN` for triplet states (bipolarons or dimers (e.g. C-C in Si apparently has ~0.5 eV energy splitting (10.1038/s41467-023-36090-2), and 0.4 eV for O-O in STO from Kanta, but smaller for VCd bipolaron in CdTe)).
  - Add our recommended  workflow (gam, NKRED, std, ncl). See https://sites.tufts.edu/andrewrosen/density-functional-theory/vasp/ for some possibly useful general tips.
  - Dielectric should be aligned with the x,y,z (or a,b,c) of the supercell right? Should check (with Kumagai), and note this in the tutorial
  - Note that bandfilling corrections are no longer supported, as in most cases they shouldn't be used anyway, and if you have band occupation in your supercell then the energies aren't accurate anyway as it's a resonant/shallow defect, and this is just lowering the energy so it sits near the band edge (leads to false charge state behaviour being a bit more common etc). If the user wants to add bandfilling corrections, they can still doing this by calculating it themselves and adding to the `corrections` attribute. (Link our code in old `pymatgen` for doing this)
  - Cite https://iopscience.iop.org/article/10.1088/1361-648X/acd3cf for validation of Voronoi tessellation
    approach for interstitials, but note user can use charge-density based approach if needing to be
    super-lean for some reason. Can use SMTG wiki stuff for this.
  - Regarding competing phases with many low-energy polymorphs from the Materials Project; will build
    in a warning when many entries for the same composition, say which have database IDs, warn the user
    and direct to relevant section on the docs -> Give some general foolproof advice for how best to deal
    with these cases (i.e. check the ICSD and online for which is actually the groundstate structure,
    and/or if it's known from other work for your chosen functional etc.)
  - Add notes about polaron finding (use SnB and/or MAGMOMs. Any other advice to add? See Abdullah/Dan chat and YouTube tutorial, should have note about setting `MAGMOM`s for defects somewhere). `doped` can't do automatically because far too much defect/material-specific dependence.
  - Show our workflow for calculating interstitials (i.e. `vasp_gam` neutral relaxations first (can point to defects tutorial for this)), and why this is recommended over the charge density method etc.
  - Add mini-example of calculating the dielectric constant (plus convergence testing with `vaspup2.0`) to docs/examples, and link this when `dielectric` used in parsing examples.
  - Note about cost of `vasp_ncl` chemical potential calculations for metals, use `ISMEAR = -5`,
    possibly `NKRED` etc. (make a function to generate `vasp_ncl` calculation files with `ISMEAR = -5`, with option to set different kpoints) - if `ISMEAR = 0` - converged kpoints still prohibitively large, use vasp_converge_files again to check for quicker convergence with ISMEAR = -5.
  - Use `NKRED = 2` for `vasp_ncl` chempot calcs, if even kpoints and over 4. Often can't use `NKRED` with `vasp_std`, because we don't know beforehand the kpts in the IBZ (because symmetry on for `vasp_std` chempot calcs)(same goes for `EVENONLY = True`).
  - Readily-usable in conjunction with `atomate`, `AiiDA`(-defects), `CarrierCapture`, and give some
    quick examples? Add as optional dependencies.
  - Workflow diagram with: https://twitter.com/Andrew_S_Rosen/status/1678115044348039168?s=20
  - Note about `ISPIN = 1` for even no. of electrons defect species, **if you're sure there's no
    magnetic ordering!** – which you can check in the `OUTCAR` by looking at `magnetization (x)` `y`
    and `z`, and checking that everything is zero (not net magnetisation, as could have opposing spin
    bipolaron). This is automatically handled in `SnB_replace_mag.py` (to be added to ShakeNBreak) and
    will be added to `doped` VASP calc scripts.
  - Setting `LREAL = Auto` can sometimes be worth doing if you have a very large supercell for speed up, _but_ it's important to do a final calculation with `LREAL = False` for accurate energies/forces, so only do if you're a power user and have a very large supercell.
  - Show usage of `get_conv_cell_site` in notebooks/docs.
  - Note in docs that `spglib` convention used for Wyckoff labels and conventional structure definition.
    Primitive structure can change, as can supercell / supercell matrix (depending on input structure,
    `generate_supercell` etc), but conventional cell should always be the same (`spglib` convention).
  - Add examples of extending analysis with `easyunfold` and `py-sc-fermi`, and get the lads to add
    this to their docs as example use cases as well. Add our thesis sc-fermi analysis notebooks to tutorials, and example of doing the Kumagai PRX Energy carrier concentration with TLD plots (can use: https://py-sc-fermi.readthedocs.io/en/latest/tutorial.html#plot-defect-concentrations-as-a-function-of-Fermi-energy). Also include examples of extending to
    non-radiative carrier capture calcs with `CarrierCapture.jl` and `nonrad`. Show example of using
    `sumo` to get the DOS plot of a defect calc, and why this is useful.
  - Worth adding a very short example showing how to set `MAGMOM`s for AFM/FM systems (see Dan & Abdullah chat)
  - Note about SOC for chemical potential calculations ([FERE paper](https://doi.org/10.1103/PhysRevB.
    85.115104) suggests that the SOC effects on total energy cancel out for chemical potential
    calculations, but only the case when the occupation of the SOC-affected orbitals is constant
    (typically not the case)) Better to do consistently (link Emily SOC work and/or thesis).
  - Link to Irea review, saying that while spin and configurational degeneracies are accounted for
    automatically in `doped`, excited-state degeneracy (e.g. with bipolarons/dimers with single and triplet
    states) are not, so the user should manually account for this if present. Also note that
    temperature effects can be important in certain cases so see this review if that's the case.
  - Should have recommendation somewhere about open science practices. The doped defect dict and thermo jsons should always be shared in e.g. Zenodo when publishing, as contains all info on the parsed defect data in a lean format. Also using the `formation_energy_table` etc. functions for SI tables is recommended.
  - Add our general rule-of-thumbs/expectations regarding charge corrections:
    - Potential alignment terms should rarely ever be massive
    - In general, the correction terms should follow somewhat consistent trends (for a given charge state, across defects), so if you see a large outlier in the corrections, it's implying something odd is happening there. This is can be fairly easily scanned with `formation_energy_table`.
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
- Should flick through other defect codes (see
  https://shakenbreak.readthedocs.io/en/latest/Code_Compatibility.html, also `AiiDA-defects`) and see if
  there's any useful functionality we want to add!
