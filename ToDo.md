### Stuff to add:
- Function to generate `vasp_std` chemical potential relaxation files, given input converged k-points (`make_converged_kpts`, `vasp_std_chempot_relax`, `vasp_ncl_chempot`).
- Note about cost of `vasp_ncl` chemical potential calculations for metals, use `ISMEAR = -5`, possibly `NKRED` etc. (make a function to generate `vasp_ncl` calculation files with `ISMEAR = -5`, with option to set different kpoints) - if ISMEAR = 0 - converged kpoints still prohibitively large, use vasp_converge_files again to check for quicker convergence with ISMEAR = -5.
- Use `NKRED = 2` for `vasp_ncl` chempot calcs, if even kpoints and over 4. Often can't use `NKRED` with `vasp_std`, because we don't know beforehand the kpts in the IBZ (because symmetry on for `vasp_std` chempot calcs)(same goes for `EVENONLY = True`).
- Add `defects_std_to_ncl.sh` bash script to auto-generate symmetrised `KPOINTS` for SOC `vasp_ncl` run from `vasp_std` IBZKPT, and copy `vasp_std/CONTCAR` to `vasp_ncl/POSCAR`, copy `CHGCAR`, `POTCAR` over etc. (Make note about symmetrised k-points required for accurate SOC bandstructures, VASP wizardry with Chris, but not an issue for ground state energies).
- Add `chempot_std_to_ncl.sh` bash script to auto-generate non-symmetrised `KPOINTS` for SOC `vasp_ncl` run (note on editing `module load vasp` command), and copy `vasp_std/CONTCAR` to `vasp_ncl/POSCAR`, copy `CHGCAR`, `POTCAR` over etc.
- Use `UserChemPotAnalyzer` to parse chemical potential calculations
- Add `dope_stuff` examples and documentation.
- `transition_levels_table()`
- Note about SOC for chemical potential calculations (Lany says: to ‘a good approximation’, the SOC contributions to total energy can be separated into purely atomic contributions, Lany, Stevanovic and Zunger show in their FERE paper (https://doi.org/10.1103/PhysRevB.85.115104) that the SOC effects on total energy cancel out for chemical potential calculations) - But only for easy systems - better to do consistently
- `NKRED` pre-relaxing on defect structures (see jspark Slack discussion)
- Build in `emphasis` option to `formation_energy_plot`, label 0 as VBM and CBM on x-axis
- Note that if you edit the entries in a DefectPhaseDiagram after creating it, you need to `dpd.find_stable_charges()` to update the transition level map etc.
- `rattle` function and notes about finding stable ground state structures (what about the bond distortion method for finding polarons?)
- `aide` labelling of defect species in formation energy plots.
- Note about `ISPIN = 1` for even no. of electrons defect species, **if you're sure there's no magnetic ordering!
- Create defects in/near the centre of the supercell (rather than near the origin), for easier visualisation.
- Create GGA practice workflow, for people to learn how to work with doped and defect calculations
- Print Wyckoff position of proposed interstitial sites (and optional output of Wyckoff sites which are neither atomic
  nor Voronoi sites) 
- Better charge state predictor? At least print determined oxidation state ranges, and warning that you're gonna use 
these to predict defect charge states (so people can see if something off etc.)
  
- Option _not to set_ certain `INCAR` tags (like HFSCREEN and LORBIT, cause their default "None" doesn't really correspond to a certain value)
- Streamline vasp_input functions (prepare_vasp_defect_inputs and prepare_vasp_defect_dict should all be done in one)
- Check against updated PyCDT to see if any big, useful changes since we copied code.
- Generate 'molecules in a box' rather than Materials Project solid forms for O2, H2, I2, Br2 etc. gas competing phases