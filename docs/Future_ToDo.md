# `doped` Future Development WishList
## Defect calculations set up
- Defect complexes: Functionality to setup and parse calculations – can do this with new `pymatgen`
  code? Note that our defect-centring code is currently not implemented for this!
- Add input file generation for FHI-AIMs, CP2K, Quantum Espresso and CASTEP (using SnB functions),
  point to post-processing tools for these online (in docs/example notebooks, `aiida-defects` for  QE,
  https://github.com/skw32/DefectCorrectionsNotebook for AIMs...),
  and give example(s) of how to manually generate `DefectThermodynamics` and chempots from the parsed
  energies of these calculations, so the `doped` analysis tools can then be used.
  - See https://github.com/materialsproject/emmet/pull/242 for CP2k defects stuff
- For defect complexes, after electrostatics, the next biggest factor in binding energies is the stress field (right)? Then orbital effects after that.
  This means that if we have the distortion field implemented in doped, we should be able to fairly accurately and easily predict if defect complexes are likely? (Via concentrations/formation energies, charges and stress fields?) Nice use case, could mention in JOSS as possible screening application if someone wanted to use it. Deak & Gali Nature Comms (10.1038/s41467-023-36090-2) C-C in Si could be used as a nice test case (neutral so no charge effects)
- **Optical transitions:** Functions for generating input files, parsing (with GKFO correction) and
  plotting the results (i.e. configuration coordinate diagrams) of optical calculations. Needs to be at
  this point because we need relaxed structures. Sensible naming scheme. Would be useful as this is a
  workflow which ppl often mess up. Can use modified code from `config-coord-plots` (but actually to
  scale and automatically/sensibly parsed etc.)(also see `CarrierCapture` functionalities)
- `doped`/`SnB`/`easyunfold` (virtual) workshop? Just noting as a possibility, could be MCC-supported.

## Post-Processing
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
  - Should have functions to plot the Fermi-level-dependent association degrees, as in Fig 3 Krasikov JMCA 2017
  - Questions some of our typical expectations of defect behaviour! Actually defect complexes are a bit
    more common than thought.
  - Could do by using the site displacements, with atoms moving outside their vdW radius being flagged
    as (possibly) defective (this is essentially the approach implemented in ``split_vacancies`` PR)? And see if their 
    stoichiometric sum matches the expected point defect stoichiometry. Expected to match one of these transformation 
    motifs:
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
- 2D charge corrections (though field still in development realistically)
- Eigenvalue corrections for the eigenvalue plots, like shown in 10.1103/PhysRevB.109.054106?
- Use the projected eigenvalues and magnetisation to detect when localised charge is associated with d/f electrons (and/or is multi-polaronic), and warn the user that different choices of NUPDOWN, maybe MAGMOM, should be tested for these defect states? (Like for dimers)

## Docs
- LDOS plotting?
- Example parsing CCD with fully-consistent charge corrections after (link recent Kumagai paper on this: https://arxiv.org/abs/2304.01454) -- would make sense in ``CarrierCapture``/``nonrad``
