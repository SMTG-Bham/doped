# `doped` Future Development WishList
## Defect calculations set up
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
  - For defect complexes, after electrostatics, the next biggest factor in binding energies is the stress field (right)? Then orbital effects after that.
   This means that if we have the distortion field implemented in doped, we should be able to fairly accurately and easily predict if defect complexes are likely? (Via concentrations/formation energies, charges and stress fields?) Nice use case, could mention in JOSS as possible screening application if someone wanted to use it. Deak & Gali Nature Comms (10.1038/s41467-023-36090-2) C-C in Si could be used as a nice test case (neutral so no charge effects)
- CLI Functionality for core functions.
  - Could also use some of the `snb` functions to add some convenience commands which `cp CONTCAR
    POSCAR` for unconverged `vasp_gam`/`vasp_nkred_std`/`vasp_std` calculations, and copies `CONTCAR`s
    to next VASP folder when converged and submits job.
- Just something to keep in mind; new defect generation code can apparently use oxidation states from
  `defect.defect_structure` and map to defect supercell. Not in our current subclass implementation of
  `Defect`. Is this useful info?
- Ideally, one should be able to input just defect objects somewhere -> an alternative input to `DefectsGenerator`? Can input to both `DefectsGenerator` or `DefectsSet` (but it'll just send it to `DefectsGenerator` with `kwargs`).
- **Optical transitions:** Functions for generating input files, parsing (with GKFO correction) and
  plotting the results (i.e. configuration coordinate diagrams) of optical calculations. Needs to be at
  this point because we need relaxed structures. Sensible naming scheme. Would be useful as this is a
  workflow which ppl often mess up. Can use modified code from `config-coord-plots` (but actually to
  scale and automatically/sensibly parsed etc.)(also see `CarrierCapture` functionalities)
- Dielectric/kpoint-sampling weighted supercell generation? (essentially just a vectorised cost function implemented in the generation loop). Would natively optimise e.g. layered materials quite well.

## Chemical Potentials
- Overhaul chemical potentials code, dealing with all `TODO`s in that module. 
  - Particularly: About the current extrinsic chempot algorithm: "SK: I don't think this is right. Here it's just getting the extrinsic chempots at the intrinsic chempot limits, but actually it should be recalculating the chempot limits with the extrinsic competing phases, then reducing _these_ facets down to those with only one extrinsic competing phase bordering".
- Once happy all required functionality is in the new `chemical_potentials.py` code (need more rigorous tests, see original pycdt tests for this and make sure all works with new code), showcase all functionality in the example notebook, and compare with old code from `vasp.py` (to ensure all functionality present).
- Currently inputting multiple extrinsic `sub_species` will assume you are co-doping, and will output competing phases for this (e.g. K and In with BaSnO3 will output KInO2), default should not be to do this, but have an optional argument for co-doping treatment.
- Functionality to combine chemical potential limits from considering different extrinsic species, to be able to plot defect formation energies for different dopants on the same diagram.
- Should output `json` of Materials Project `ComputedStructureEntry` used for each competing phase directory, to aid provenance.
- `vasp_ncl` etc input file generation, `vaspup2.0` `input` folder with `CONFIG` generation, improve `chemical_potentials` docstrings (i.e. mention defaults, note in notebooks if changing `INCAR`/`POTCAR` settings for competing phase production calcs, should also do with defect supercell calcs / vice versa)

## Post-Processing
- Symmetry determination in arbitrary (periodicity-breaking) supercells. Should be doable, with defect-expander (stenciling) type code to regenerate the structure in an appropriate larger non-periodicity-breaking cell.
- For complex defects, auto symmetry determination is future work, and should be done manually (note in docs and give
  warning when parsing).
  - Previously had ideas (in `Dev_ToDo.md`) about split-interstitials/vacancies, but think these are now handled fine with current tools.
- Parsing capability for (non-defect) polarons, so they can then be plotted alongside defects on
  formation energy diagrams. Main things for this are:
  - Input file generation
  - Parsing to determine polaron site (so we can then use charge corrections). Use the site of max
    displacement / bond length difference for this, and future work could be parsing of charge densities
    to get the maximum position. (Note in docs that the user can do this if they want it).
  - General plotting (in transition level diagrams) and analysis (e.g. our site displacement/strain
    functions).
- Complex defect / defect cluster automatic handling. Means we can natively handle complex defects, and
  also important for e.g. `ShakeNBreak` parsing, as in many cases we're ending up with what are
  effectively defect clusters rather than point defects (e.g. V_Sb^+1 actually Se_Sb^-1 + V_Se^+2 in
  Xinwei's https://doi.org/10.1103/PhysRevB.108.134102), so it would be really nice to have this automatic parsing
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
- `doped` functions should be equally applicable to the base `pymatgen` `Defect`/`DefectEntry` objects (as well as the corresponding `doped` subclasses) as much as possible. Can we add some quick tests for this? 
- Implement shallow donor/acceptor binding estimation functions (via effective mass theory)
- Kasamatsu formula for defect concentrations at high concentrations (accounts for lattice site competition), as shown in DefAP paper

## Docs
- Add LDOS plotting, big selling point for defects and disorder!
- Add short example notebook showing how to generate a defect PES / NEB and then parse with fully-consistent charge corrections after (link recent Kumagai paper on this: https://arxiv.org/abs/2304.01454). SK has the code for this in local example notebooks ready to go.

