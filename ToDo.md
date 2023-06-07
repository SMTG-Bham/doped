# Stuff to add
## Chemical potential
- Update chemical potential tools to work with new Materials Project API. Currently, supplying an API key for the new Materials Project API returns entries which do not have `e_above_hull` as a property, and so crashes. Ideally would be good to be compatible with both the legacy and new API, which should be fairly straightforward (try importing MPRester from mp_api client except ImportError import from pmg then will need to make a whole separate query/search because `band_gap` and `total_magnetisation` no longer accessible from `get_entries`). See https://docs.materialsproject.org/downloading-data/using-the-api
- Currently inputting multiple extrinsic `sub_species` will assume you are co-doping, and will output competing phases for this (e.g. K and In with BaSnO3 will output KInO2), default should not be to do this, but have an optional argument for co-doping treatment.
- Note about cost of `vasp_ncl` chemical potential calculations for metals, use `ISMEAR = -5`, possibly `NKRED` etc. (make a function to generate `vasp_ncl` calculation files with `ISMEAR = -5`, with option to set different kpoints) - if `ISMEAR = 0` - converged kpoints still prohibitively large, use vasp_converge_files again to check for quicker convergence with ISMEAR = -5.
- Use `NKRED = 2` for `vasp_ncl` chempot calcs, if even kpoints and over 4. Often can't use `NKRED` with `vasp_std`, because we don't know beforehand the kpts in the IBZ (because symmetry on for `vasp_std` chempot calcs)(same goes for `EVENONLY = True`).
- Add `chempot_std_to_ncl.sh` bash script to auto-generate symmetrised `KPOINTS` for SOC `vasp_ncl` run from `vasp_std` IBZKPT, and copy `vasp_std/CONTCAR` to `vasp_ncl/POSCAR`, copy `CHGCAR`, `POTCAR` over etc. (Make note about symmetrised k-points required for accurate SOC bandstructures, VASP wizardry with Chris, but not an issue for ground state energies).
- Note about SOC for chemical potential calculations (Lany says: to ‘a good approximation’, the SOC contributions to total energy can be separated into purely atomic contributions, Lany, Stevanovic and Zunger show in their [FERE paper](https://doi.org/10.1103/PhysRevB.85.115104) that the SOC effects on total energy cancel out for chemical potential calculations) - But only for easy systems - better to do consistently
- Publication ready chemical potential diagram plotting tool (see `doped_chempot_plotting_example.ipynb`; code there, just needs to be implemented in module functions).
- Functionality to combine chemical potential limits from considering different extrinsic species, to be able to plot defect formation energies for different dopants on the same diagram.
- Functionality to generate chemical potential limit plots from parsed chempot calculations (phase diagram objects), as in Adam Jackson's `plot-cplap-ternary` (3D) and Sungyhun's `cplapy` (4D). – See `Cs2SnTiI6` notebooks for template code for this.
- Once happy all required functionality is in the new `chemical_potentials.py` code (need more rigorous tests, see original pycdt tests for this and make sure all works with new code), showcase all functionality in the example notebook, remove the old modified-pycdt `_chemical_potentials.py` code.

## Defect calculations set up
- Defect complexes: Functionality to setup and parse calculations – can do this with new `pymatgen` code?
- Better charge state predictor? At least print determined oxidation state ranges, and warning that you're gonna use these to predict defect charge states (so people can see if something off etc.); could use the csv Dan sent on defects slack (17 Mar 21 - this can also be done in pymatgen; see ShakeNBreak most_common_oxi function) and set an arbitrary cutoff for oxidation states that can occur in known materials. Alternative possibility is do +/-2 to fully-ionised+/-2, as this should cover >99% of amphoteric cases right? (See emails with Jimmy – can be easily done with 'padding' option in pymatgen-analysis-defects?)
- Add function to post-process and remove closely-located interstitials for structures with large voids (from SMTG #software Slack (Yong-Seok): "If your structure has large space for interstitials and it predicts lots of atoms closely positioned to each other (& take longer time to predict), you can increase min_dist  (default is 0.5) in remove_collisions function in [python path]/python3.9/site-packages/pymatgen/analysis/defects/utils.py"), and add note to example notebooks about this.
- Functions for generating input files, parsing (with GKFO correction) and plotting the results (i.e. configuration coordinate diagrams) of optical calculations. Integrate with Joe's `config-coord-plots`? (also see `CarrierCapture` functionalities)
- Add defect expansion code functionality to regenerate defect structures from a smaller supercell in a larger one. Useful for supercell size convergence tests, and accelerating `ShakeNBreak` etc. If/when adding, make sure to link in `SnB` docs as well.
  - Related point, using our `doped` site-matching functions, could write some quick functions to plot the exponential tailing off of strain / site displacements as we move away from the defect site. Could be useful as a validation / check of supercell size convergence, and for quantifying the strain / distortion introduced by a certain defect (though I guess the `SnB` tools already do a good job of that) – could possibly give a good rule-of-thumb to aim for with a sufficiently large cell?

## Post-processing / analysis / plotting
- Automatically check the 'bulk' and 'defect' calculations used the same INCAR tags, KPOINTS and POTCAR settings, and warn user if not.
- Better automatic defect formation energy plot colour handling (auto-change colormap based on number of defects, set similar colours for similar defects (types and inequivalent sites)) – and more customisable?
  - Ordering of defects plotted (and thus in the legend) should be physically relevant (whether by energy, or defect type etc.)
  - Should have `ncols` as an optional parameter for the function, and auto-set this to 2 if the legend height exceeds that of the plot
  - Don't show transition levels outside of the bandgap (or within a certain range of the band edge, possibly using `pydefect` delocalisation analysis?), as these are shallow and not calculable with the standard supercell approach.
  - Use the update defect name info in `plotting` plotting? i.e. Legend with the inequivalent site naming used in the subscripts?
- Add LDOS plotting to doped, big selling point for defects and disorder!
- Add short example notebook showing how to generate a defect PES / NEB and then parse with fully-consistent charge corrections after (link recent Kumagai paper on this: https://arxiv.org/abs/2304.01454). SK has the code for this in local example notebooks ready to go.
- `aide` labelling of defect species in formation energy plots?
- Note that if you edit the entries in a DefectPhaseDiagram after creating it, you need to `dpd.find_stable_charges()` to update the transition level map etc.
- `transition_levels_table()`. Also ensure we have functionality to print all single-electron TLs (useful to know when deciding what TLs to do carrier capture for. @SeánK has code for this in jupyter notebooks)
- Change formation energy plotting and tabulation to DefectPhaseDiagram methods rather than standalone functions.
- Add warning for bandfilling correction based off energy range of the CBM/VBM occupation? (In addition to num_hole and num_electron)
- Option for degeneracy-weighted ('reduced') formation energy diagrams, similar to reduced energies in SOD. See Slack discussion and CdTe pyscfermi notebooks.
- Improved methods for estimating/determining the final site degeneracy/multiplicity from relaxed structures. See `pydefect` for tools for this. Should be doable with current point symmetry tools, especially when both the defect and bulk structures are available. Also add consideration of odd/even number of electrons to account for spin degeneracy.
- Brouwer diagrams. Also see Fig. 6a of the `AiiDA-defects` preprint, want plotting tools like this (some could be PR'd to `py-sc-fermi`)
- Function(s) for exporting defect energies and corrections as Pandas DataFrame / HDF5 / json / yaml / csv etc for readily-accessible, easy-to-use reproducibility
- Functions to output data and python objects to plug and play with `py-sc-fermi`, `AiiDA`, `CarrierCapture`. Seán K has functions and notebooks for transferring to `py-sc-fermi` and generating nice plots with the outputs, so will add this.
- Parsing capability for (non-defect) polarons, so they can then be plotted alongside defects on formation energy diagrams.
- Add warning if, when parsing, only one charge state for a defect is parsed (i.e. the other charge states haven't completed), in case this isn't noticed by the user. Print a list of all parsed charge states as a check.
- Improved handling of the delocalisation analysis warning. `pymatgen`'s version is too sensitive. Maybe if `pymatgen` finds the defect to be incompatible, estimate the error in the energy, and if small enough ignore, otherwise give an informative warning of the estimated error, possible origins (unreasonable/unstable/shallow charge state, as the charge is being significantly delocalised across the cell, rather than localised at the defect) – this has been tanked in new `pymatgen`. Could just use the `pydefect` shallow defect analysis instead?
- Currently the `PointDefectComparator` object from `pymatgen.analysis.defects.thermodynamics` is used to group defect charge states for the transition level plot / transition level map outputs. For interstitials, if the closest Voronoi site from the relaxed structure thus differs significantly between charge states, this will give separate lines for each charge state. This is kind of ok, because they _are_ actually different defect sites, but should have intelligent defaults for dealing with this (see `TODO` in `dpd_from_defect_dict` in `analysis.py`; at least similar colours for similar defect types, an option to just show amalgamated lowest energy charge states for each _defect type_). NaP is an example for this – should have a test built for however we want to handle cases like this. See Ke's example case too with different interstitial sites.
- `pydefect` integration, so we can use:
  - Handling of shallow defects
  - Readily automated with `vise` if one wants (easy high-throughput and can setup primitive calcs (BS, DOS, dielectric).
  - Some nice defect structure and eigenvalue analysis
  - GKFO correction

## Housekeeping
- Logo!
- Clean `README` with bullet-point summary of key features, and sidebar like `SnB`.
- Update to be compatible with new `pymatgen`
  - Update to use the `ShakeNBreak` voronoi node-finding functions, as this has been made to be more efficient than the `doped` version (which is already far more efficient than the original...) and isn't available in current `pymatgen`.
- Create GGA practice workflow, for people to learn how to work with doped and defect calculations
- Test coverage.
- PR to pymatgen: Update entry.parameters["kumagai_meta"] = (dict(self.metadata)) to entry.parameters["kumagai_meta"].update(dict(self.metadata)) in KumagaiCorrection.get_correction() in pymatgen/analysis/defects/corrections.py so pymatgen doesn't remove the other relevant kumagai_meta (kumagai_electrostatic etc.) when we run KumagaiCorrection.get_correction(defect_entry) (via finite_size_charge_correction.get_correction_kumagai(defect_entry...)) – see https://github.com/materialsproject/pymatgen-analysis-defects/issues/47 – code now gone, so can we add a workaround to `corrections.get_correction_kumagai()` for this?
- Generate docs.
  - Add note about `NUPDOWN` for triplet states (bipolarons).
  - Add our recommended  workflow (gam, NKRED, std, ncl). Cite https://iopscience.iop.org/article/10.1088/1361-648X/acd3cf for validation of Voronoi tessellation approach for interstitials, but note user can use charge-density based approach if needing to be super-lean for some reason.
  - Add notes about polaron finding (use SnB or MAGMOMs. Any other advice to add?)
  - Show our workflow for calculating interstitials (i.e. `vasp_gam` neutral relaxations first (can point to defects tutorial for this)), and why this is recommended over the charge density method etc.
  - Add mini-example of calculating the dielectric constant (plus convergence testing with `vaspup2.0`) to docs/examples, and link this when `dielectric` used in parsing examples.
  - Readily-usable in conjunction with `atomate`, `AiiDA`, `CarrierCapture`, and give some examples.
  - Note about `ISPIN = 1` for even no. of electrons defect species, **if you're sure there's no
    magnetic ordering!** – which you can check in the `OUTCAR` by looking at `magnetization (x)` `y`
    and `z`, and checking that everything is zero (not net magnetisation, as could have opposing spin
    bipolaron). This is automatically handled in `SnB_replace_mag.py` (to be added to ShakeNBreak) and
    will be added to `doped` VASP calc scripts.
