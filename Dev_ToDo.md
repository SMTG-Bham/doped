# `doped` Development To-Do List
## Defect calculations set up
- See SK Remarkable notes

## Chemical potential
- Update chemical potential tools to work with new Materials Project API. Currently, supplying an API key for the new Materials Project API returns entries which do not have `e_above_hull` as a property, and so crashes. Ideally would be good to be compatible with both the legacy and new API, which should be fairly straightforward (try importing MPRester from mp_api client except ImportError import from pmg then will need to make a whole separate query/search because `band_gap` and `total_magnetisation` no longer accessible from `get_entries`). See https://docs.materialsproject.org/downloading-data/using-the-api
- Currently inputting multiple extrinsic `sub_species` will assume you are co-doping, and will output competing phases for this (e.g. K and In with BaSnO3 will output KInO2), default should not be to do this, but have an optional argument for co-doping treatment.
- Publication ready chemical potential diagram plotting tool as in Adam Jackson's `plot-cplap-ternary` (3D) and Sungyhun's `cplapy` (4D) (see `doped_chempot_plotting_example.ipynb`; code there, just needs to be implemented in module functions).
  - Also see `Cs2SnTiI6` notebooks for template code for this.
- Functionality to combine chemical potential limits from considering different extrinsic species, to be able to plot defect formation energies for different dopants on the same diagram.
- Once happy all required functionality is in the new `chemical_potentials.py` code (need more rigorous tests, see original pycdt tests for this and make sure all works with new code), showcase all functionality in the example notebook, remove the old modified-pycdt `_chemical_potentials.py` code.

## Post-processing / analysis / plotting
- Automatically check the 'bulk' and 'defect' calculations used the same INCAR tags, KPOINTS and POTCAR
  settings, and warn user if not. Should auto-check the magnetisation output; if it comes to around
  zero for an odd-electron defect, suggests getting spurious shallow defect behaviour!
- Profile defect parsing, identify bottlenecks and consider if multiprocessing could be used to speed up.
- Add warning if, when parsing, only one charge state for a defect is parsed (i.e. the other charge
  states haven't completed), in case this isn't noticed by the user. Print a list of all parsed charge
  states as a check.
- Try re-determine defect symmetry and site multiplicity (particularly important for interstitials, as
  relaxation may move them to lower/higher symmetry sites which significantly different multiplicity).
  - Should be doable with current point symmetry tools, especially when both the defect and bulk
    structures are available. The configurational degeneracy should be just the final site degeneracy
    (i.e. Wyckoff number) divided by the initial, or equivalently the initial number of symmetry
    operations divided by the final, so we can just use this to determine the final site degeneracies.
    For interstitials, should be based off just the Wyckoff number of the final relaxed site.
    Should make this a parsed defect property, defined relative to the conventional cell (so they
    actually correspond to Wyckoff numbers, will need some idiotproof checks/notes for users about this),
    and have this automatically plug-and-play with `py-sc-fermi`. Already have the site analysis /
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
  - Also add consideration of odd/even number of electrons to account for spin degeneracy.
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
- Previous `pymatgen` issues, fixed?
  - Improved handling of the delocalisation analysis warning. `pymatgen`'s version is too sensitive. Maybe if `pymatgen` finds the defect to be incompatible, estimate the error in the energy, and if small enough ignore, otherwise give an informative warning of the estimated error, possible origins (unreasonable/unstable/shallow charge state, as the charge is being significantly delocalised across the cell, rather than localised at the defect) – this has been tanked in new `pymatgen`. Could just use the `pydefect` shallow defect analysis instead?
  - Related: Add warning for bandfilling correction based off energy range of the CBM/VBM occupation? (In
    addition to `num_hole` and `num_electron`)
  - Currently the `PointDefectComparator` object from `pymatgen.analysis.defects.thermodynamics` is used to group defect charge states for the transition level plot / transition level map outputs. For interstitials, if the closest Voronoi site from the relaxed structure thus differs significantly between charge states, this will give separate lines for each charge state. This is kind of ok, because they _are_ actually different defect sites, but should have intelligent defaults for dealing with this (see `TODO` in `dpd_from_defect_dict` in `analysis.py`; at least similar colours for similar defect types, an option to just show amalgamated lowest energy charge states for each _defect type_). NaP is an example for this – should have a test built for however we want to handle cases like this. See Ke's example case too with different interstitial sites.
  - GitHub issue related to `DefectPhaseDiagram`: https://github.com/SMTG-UCL/doped/issues/3 -> Think about how we want to refactor the `DefectPhaseDiagram` object!
  - Note that if you edit the entries in a DefectPhaseDiagram after creating it, you need to `dpd.find_stable_charges()` to update the transition level map etc.
- Should tag parsed defects with `is_shallow` (or similar), and then omit these from plotting/analysis
  (and note this behaviour in examples/docs)
- Ideally our defect parsing would be able to get the final _relaxed_ position of vacancies / antisites that move significantly (or the centroid if a defect cluster), to then use for the charge correction. Not a big deal for larger supercells, but a slight mismatch in defect site prediction for smaller supercells can have a semi-significant effect on the predicted charge correction. `Int_Te_3_unperturbed_1` is a good example of this tricky case.
- Change formation energy plotting and tabulation to DefectPhaseDiagram methods rather than standalone
  functions – with `pymatgen` update what's the new architecture?
- Better automatic defect formation energy plot colour handling (auto-change colormap based on number of defects, set similar colours for similar defects (types and inequivalent sites)) – and more customisable?
  - `aide` labelling of defect species in formation energy plots?
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
- Brouwer diagrams. Also see Fig. 6a of the `AiiDA-defects` preprint, want plotting tools like this (some could be PR'd to `py-sc-fermi`)
- Function(s) for exporting defect energies and corrections as Pandas DataFrame / HDF5 / json / yaml / csv etc for readily-accessible, easy-to-use reproducibility
- Functions to output data and python objects to plug and play with `py-sc-fermi`, `AiiDA`, `CarrierCapture`.
  - Alex Squires has functions/notebooks from Seán (-> Xinwei -> Jiayi) -> Alex, for transferring to
    `py-sc-fermi` and generating nice plots with the outputs, so add this and make it our suggested
    workflow in the docs etc.
  - `py-sc-fermi` may have functionality for dealing with complex defect concentrations in the future
    (see Slack with Alex; 07/06/23)
- `pydefect` integration, so we can use:
  - Handling of shallow defects
  - Readily automated with `vise` if one wants (easy high-throughput and can setup primitive calcs (BS, DOS, dielectric).
  - Some nice defect structure and eigenvalue analysis
  - GKFO correction

## Housekeeping
- Clean `README` with bullet-point summary of key features, and sidebar like `SnB`.
- `ShakeNBreak` related updates:
  - Use doped naming conventions and functions and defect entry generation functions in `ShakeNBreak`.
- Code tidy up:
  - Notebooks in `tests`; update or delete.
  - Test coverage?
  - Go through docstrings and trim to 80 characters. Also make sure all tidy and understandable (idiot-proof). Should be able to generate docs with little to no warnings/errors.
  - Add type hints for all functions.
  - Check code for each function is relatively lean & efficient, can check with ChatGPT for any gnarly
    ones.

- Docs:
  - Create GGA practice workflow, for people to learn how to work with doped and defect calculations
  - Add note about `NUPDOWN` for triplet states (bipolarons or dimers (e.g. C-C in Si apparently has ~0.5 eV energy splitting (10.1038/s41467-023-36090-2), and O-O in STO from Kanta?)).
  - Add our recommended  workflow (gam, NKRED, std, ncl). See https://sites.tufts.edu/andrewrosen/density-functional-theory/vasp/ for some possibly useful general tips.
  - Cite https://iopscience.iop.org/article/10.1088/1361-648X/acd3cf for validation of Voronoi tessellation
    approach for interstitials, but note user can use charge-density based approach if needing to be
    super-lean for some reason. Can use SMTG wiki stuff for this.
  - Regarding competing phases with many low-energy polymorphs from the Materials Project; will build
    in a warning when many entries for the same composition, say which have database IDs, warn the user
    and direct to relevant section on the docs -> Give some general foolproof advice for how best to deal
    with these cases (i.e. check the ICSD and online for which is actually the groundstate structure,
    and/or if it's known from other work for your chosen functional etc.)
  - Add notes about polaron finding (use SnB or MAGMOMs. Any other advice to add?)
  - Show our workflow for calculating interstitials (i.e. `vasp_gam` neutral relaxations first (can point to defects tutorial for this)), and why this is recommended over the charge density method etc.
  - Add mini-example of calculating the dielectric constant (plus convergence testing with `vaspup2.0`) to docs/examples, and link this when `dielectric` used in parsing examples.
  - Note about cost of `vasp_ncl` chemical potential calculations for metals, use `ISMEAR = -5`,
    possibly `NKRED` etc. (make a function to generate `vasp_ncl` calculation files with `ISMEAR = -5`, with option to set different kpoints) - if `ISMEAR = 0` - converged kpoints still prohibitively large, use vasp_converge_files again to check for quicker convergence with ISMEAR = -5.
  - Use `NKRED = 2` for `vasp_ncl` chempot calcs, if even kpoints and over 4. Often can't use `NKRED` with `vasp_std`, because we don't know beforehand the kpts in the IBZ (because symmetry on for `vasp_std` chempot calcs)(same goes for `EVENONLY = True`).
  - Readily-usable in conjunction with `atomate`, `AiiDA`(-defects), `CarrierCapture`, and give some
    examples. Add as optional dependencies.
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
    this to their docs as example use cases as well. Add our thesis sc-fermi analysis notebooks to tutorials. Also include examples of extending to
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
  - The Wyckoff analysis code is very useful and no other package can do this afaik. See
    https://github.com/spglib/spglib/issues/135. Should describe and exemplify this in the docs (i.e. the
    `get_wyckoff_label_and_equiv_coord_list()` from just a `pymatgen` site and spacegroup 🔥) and JOSS
    paper.
  - Note that charge states are guessed based on different factors, but these rely on auto-determined
    oxidation states and can fail in weird cases. As always please consider if these charge states are
    reasonable for the defects in your system. (i.e. low-symmetry, amphoteric, mixed-valence cases etc!)
  - Show quick example case of the IPR code from `pymatgen-analysis-defects` (or from Adair code? or others?)
- Should flick through other defect codes (see
  https://shakenbreak.readthedocs.io/en/latest/Code_Compatibility.html, also `AiiDA-defects`) and see if
  there's any useful functionality we want to add!
