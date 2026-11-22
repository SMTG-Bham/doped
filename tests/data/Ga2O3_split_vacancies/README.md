# V_Ga(-3) point and split vacancies in β-Ga₂O₃, across supercell sizes

Test data for the multi-site (`point_charges`) finite-size charge corrections, from real DFT relaxations
of Ga vacancies (q = -3) in β-Ga₂O₃, performed as part of https://arxiv.org/abs/2611.19365.

`Ga2O3_split_vacancies_defect_dict.json.gz` is a dictionary of `DefectEntry` objects, keyed
`<supercell>/<name>`, for 8 configurations in 12 supercells (160-1280 atoms; 69 calculations):

- `v_Ga1`, `v_Ga2`: point vacancies (tetrahedral Ga(I), octahedral Ga(II));
- `v_Ga_2split_ic` (the DFT ground state), `v_Ga_2split_ib`: 2 V_Ga + 1 Ga_i;
- `v_Ga_3split_68`, `v_Ga_3split_27`: 3 V_Ga + 2 Ga_i;
- `v_Ga_4split_119`, `v_Ga_4split_218`: 4 V_Ga + 3 Ga_i (extending up to 8.1 Å).

Supercells are `N_a x N_b x N_c` expansions of the conventional cell (a = 12.28, b = 3.05, c = 5.82 Å),
some with sheared lattice vectors; all share one Cartesian frame (x ∥ b, y ∥ a, z ∥ c*). The point
vacancies and `v_Ga_2split_ib` / `v_Ga_3split_68` are only in the 7 smallest cells; `v_Ga_2split_ic` is in
all 12.

Each entry holds what the eFNV and FNV corrections need, so no `OUTCAR` or `LOCPOT` files are required:

- `sc_entry` / `bulk_entry`: relaxed defect and bulk supercells with their energies;
- `calculation_metadata["defect_site_potentials"]` / `["bulk_site_potentials"]`: atomic core potentials;
- `calculation_metadata["defect_locpot_dict"]` / `["bulk_locpot_dict"]`: the planar-averaged electrostatic
  potentials along each lattice vector, from `LVHAR` Γ-only single points on the relaxed structures (which
  match the relaxations' structures and energies to < 1e-4 Å and < 1 meV), rounded to 1 µV (changing the FNV
  corrections by < 1e-6 eV);
- `calculation_metadata["point_charges"]`: the multi-site charge model, as `(charge, frac_coords)` with
  -3 on each vacant Ga site and +3 on each Ga interstitial (summing to -3), unwrapped to one periodic image.
  The sites are from the dataset's own classification (atoms within 0.6 Å of a bulk site occupy it), with
  the Ga_i sites drift-corrected (shifted by minus the mean displacement of the other atoms, 0.001-0.022 Å),
  and otherwise match the sites identified by `doped` (`get_defect_type_and_site_indices` with `site_tol`);
- `calculation_metadata["dielectric"]`: the PBEsol DFPT static dielectric tensor (ionic + electronic; `IBRION = 8`
  + `LEPSILON` with local-field effects, 500 eV, Γ-centred 11×6×6, on the 10-atom primitive cell of the relaxed
  bulk), in the supercells' Cartesian frame: `[[10.89, 0, 0], [0, 10.98, 0.78], [0, 0.78, 13.49]]`. With this
  sign of the off-diagonal element (which depends on the sense of z), the values along a*, b and c* (10.76,
  10.89, 13.49) follow the measured ordering (10.2, 10.87, 12.4; Fiedler *et al.*, *ECS J. Solid State Sci.
  Technol.* **8**, Q3083 (2019))*.

`corrections` is empty (and `defect_supercell` / `bulk_supercell` are unset, to halve the file size).
The `defect`, defect site and symmetry attributes of each entry are from `doped`'s (single vacancy) parsing, so
are not meaningful for the split configurations.

## Caveats

- **Γ-only energies.** Total energies are from Γ-point relaxations, which are not fully converged for the
  smaller cells, and the Γ-folded bulk VBM differs by up to 24 meV between cells.
- Only the planar averages of the `LOCPOT`s are included, so corrections needing the 3D potential (e.g.
  `sxdefectalign`) cannot be computed from this data. The source dataset's 100-atom cell (`99atom`) is not included.

## DFT settings

VASP 6.5.1, PBEsol, PAW_PBE Ga_d / O, 500 eV, Γ-only, fixed cell, EDIFFG -0.01 eV/Å; neutral bulk
references with the same settings.

## Generation

Each calculation was parsed with `DefectParser.from_paths(defect_path, bulk_path, dielectric=...)`, then
`run_metadata`, the unrelaxed/guessed-initial structures and the parse-time corrections were removed, and
`point_charges` added from the dataset's `sites.json` (`CONTCAR_relaxed` classification), unwrapped with
`doped.utils.supercells._get_unwrapped_complex_frac_coords`. The planar averages were added from the dataset's
`gamma_planar_avg.tar` (`{str(i): Locpot.get_average_along_axis(i) for i in range(3)}` of each `LVHAR` `LOCPOT`).
