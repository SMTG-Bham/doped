---
orphan: true
---

# Known MP Ground States
This document lists some known ground states / room-temperature phases for common elements and compounds 
in the Materials Project (MP) database, which can be used to reduce the number of calculations required 
for chemical potential (competing phase) and phase diagram calculations. Citations are given for the 
studies where the hybrid DFT energy analyses (of candidate low-energy polymorphs) were initially performed. 

## Elemental Phases
### Chalcogens (S, Se, Te)
#### Sulfur (S)
- Sulfur is a solid at ambient (room) temperature and pressure, formed from packing S8 (cyclooctasulfur) 
  molecules in a `Fddd1` spacegroup structure. This corresponds to the lowest energy sulfur phase on the 
  MP database (`mp-77`; EaH = Energy above Hull = 0 eV/atom in the MP database).
- Gas phase S8 (cyclooctasulfur), with isolated S8 molecules, corresponds to `mp-994911` 
  (EaH ~ 0.042 eV/atom in the MP database), typically being the dominant allotrope in gas/vapour phase 
  sulfur at low-to-moderate temperatures (400 K ~< T ~< 1000 K).
- S2 (disulfur) is the dominant sulfur vapour phase at high temperatures (T ~> 1000 K). `mp-1064933` is a 
  close-packed variant of S2, or the standard molecular phase can be generated as a _molecule in a box_ 
  with the `doped.chemical_potentials.make_molecule_in_a_box` function.
Citation: https://doi.org/10.1038/s41467-022-32669-3

There are many many low energy allotropes of sulfur on the MP database (and indeed in reality), but 
typically only these phases are relevant for competing phase calculations and chemical potentials.

Hybrid (HSE06) DFT has been found to give consistent energy ordering with MP energies, with `mp-77` 
(close-packed S8) being the lowest energy sulfur allotrope (polymorph) in static athermal calculations. 
Sulfur is a solid at room temperature, but becomes a liquid/vapour phase at higher temperatures (~> 400 K), 
and so for accurate chemical potentials (and thus defect formation energies) at high temperatures, it is 
recommended to account for the vapour phase temperature dependence; whether through direct DFT 
calculations (combining ideal gas laws and vibrational free energies), through parametrised models (e.g.
https://doi.org/10.1039/C5SC03088A, https://doi.org/10.1021/acsaem.3c03208) or otherwise.

#### Selenium (Se)
- Selenium is a solid at ambient (room) temperature and pressure, adopting the trigonal P3₁21 phase 
  (`mp-14`, EaH ~ 0.001 eV/atom in the MP database).
- Under athermal conditions, both MP calculations and hybrid (HSE06+D3) DFT agree that 'red selenium' 
  (γ-monoclinic P2₁/c) is the lowest energy selenium phase (`mp-570481`; EaH = 0 eV/atom in the MP 
  database), as discussed in https://doi.org/10.1039/D4EE04647A (Table 2 SI). This phase corresponds to a packing of Se8 molecules in a `P2₁/c` spacegroup structure, similar to the solid sulfur groundstate.
Citation: https://doi.org/10.1039/D4EE04647A

There are several low energy allotropes of selenium on the MP database (and indeed in reality), but 
typically only these phases are relevant for competing phase calculations and chemical potentials.

#### Tellurium (Te)
- Under standard conditions, tellurium adopts the trigonal P3₁21 phase (`mp-19`, EaH = 0 eV/atom in MP 
  database). Hybrid (HSE06) DFT has been found to give consistent energy ordering with MP energies, with 
  `mp-19` being the lowest energy tellurium allotrope (polymorph) in static athermal calculations.
Citation: https://discovery.ucl.ac.uk/id/eprint/10186130/

There are several low energy allotropes of tellurium on the MP database (and indeed in reality), but 
typically only this phase is relevant for competing phase calculations and chemical potentials.

### Alkali Metals (Li, Na, K, Rb, Cs)
#### Lithium (Li)
- `mp-51` is the true FCC `Fm-3m` ground-state structure of lithium at low temperatures (T ~< 70 K), 
  correctly predicted as the lowest energy lithium phase in static athermal calculations with hybrid DFT
  (HSE w/34.5% exchange), and the 2nd-lowest Li phase on the MP database (EaH ~ 0.0025 eV/atom in the MP database).
- `mp-136` is the BCC `Im-3m` lithium phase which is stable at higher temperatures (T ~> 70 K, and thus 
  room temperature); EaH ~ 0.01 eV/atom in the MP database.
Citation: https://discovery.ucl.ac.uk/id/eprint/10186130/

#### Sodium (Na)
- `mp-127` is the true BCC `Im-3m` ground-state structure of sodium, though with EaH ~ 0.016 eV/atom in MP
  database.
- Hybrid (HSE06) DFT predicts the `mp-10172` `P6_3/mmc` phase as the lowest energy sodium allotrope in 
  static athermal calculations, in agreement with MP database energies (EaH ~ 0 eV/atom in the MP 
  database).
Citation: https://doi.org/10.1038/s41467-022-32669-3r

#### Potassium (K)
- `mp-58` is the true BCC `Im-3m` ground-state structure of potassium, though with EaH ~ 0.03 eV/atom in the 
  MP database.
- Hybrid DFT (HSE w/34.5% exchange) predicts the `mp-604325` `C2/c` phase as the lowest energy potassium
  allotrope in static athermal calculations (EaH ~ 0.07 eV/atom in the MP database), ~0.01 eV/atom lower energy
  than `mp-58`.
Citation: https://discovery.ucl.ac.uk/id/eprint/10186130/

#### Rubidium (Rb)
- `mp-70` is the true BCC `Im-3m` ground-state structure of rubidium, though with EaH ~ 0.009 eV/atom in 
  the MP database.
- Hybrid DFT (HSE w/34.5% exchange) predicts the `mp-656615` `P1` phase as the lowest energy rubidium
  allotrope in static athermal calculations (EaH ~ 0.023 eV/atom in the MP database), only slightly (~0.005 eV/
  atom) lower energy than `mp-70`.
Citation: https://discovery.ucl.ac.uk/id/eprint/10186130/

#### Caesium (Cs)
- `mp-1` is the true BCC `Im-3m` ground-state structure of caesium, though with EaH ~ 0.02 eV/atom in the 
  MP database.
- Hybrid (HSE06) DFT predicts the `mp-1184151` `I-43m` phase (EaH ~ 0.013 eV/atom in the MP database) as the
  lowest energy caesium phase in static athermal calculations.
Citation: https://doi.org/10.1021/acs.jpcc.3c05204

## Pnictogens (P, As, Sb, Bi)
#### Phosphorus (P)
- `mp-157` is black phosphorus, the true orthorhombic `Cmce` ground-state structure of phosphorus, though
  with EaH ~ 0.02 eV/atom in the MP database (which does not include dispersion corrections, expected to 
  stabilise this phase).
- `mp-1198724` is a red phosphorus type phase, being the lowest energy phosphorus phase on the MP database
  (EaH = 0 eV/atom), and the lowest energy phase in static athermal calculations with hybrid DFT (HSE06), 
  _without dispersion corrections_.
Citation: https://discovery.ucl.ac.uk/id/eprint/10186130/; https://doi.org/10.1103/PRXEnergy.2.043002

#### Arsenic (As)
- `mp-11` is the true rhombohedral (trigonal) `R-3m` ground-state structure of arsenic, though with EaH ~ 
  0.011 eV/atom in the MP database. Hybrid DFT (HSE w/34.5% exchange) correctly predicts this phase as the 
  lowest energy arsenic phase in static athermal calculations.
Citation: https://discovery.ucl.ac.uk/id/eprint/10186130/

#### Antimony (Sb)
- `mp-104` is the true rhombohedral (trigonal) `R-3m` ground-state structure of antimony, correctly 
  predicted as the lowest energy antimony phase in the MP database (EaH = 0 eV/atom) and in static 
  athermal calculations with hybrid DFT (HSE w/34.5% exchange).
Citation: https://discovery.ucl.ac.uk/id/eprint/10186130/

#### Bismuth (Bi)
- `mp-23152` is the true rhombohedral (trigonal) `R-3m` ground-state structure of bismuth, which both MP
  (GGA) database and hybrid (HSE06) DFT energies predict as the lowest energy bismuth allotrope in static
  athermal calculations.
Citation: https://doi.org/10.1038/s41467-022-32669-3r

## Halogens (Br, I)
#### Bromine (Br)
- `mp-23154` is the true orthorhombic `Cmce`/`Cmca` ground-state structure of bromine under ambient 
  and low temperature conditions, which both MP (GGA) database and hybrid (HSE06) DFT energies 
  correctly predict in static athermal calculations.

Citation: https://doi.org/10.1021/acs.jpcc.3c05204

#### Iodine (I)
- `mp-1525634` is the true orthorhombic `Cmce` ground-state structure of iodine, correctly predicted as the 
  lowest energy iodine phase in the MP database (EaH = 0 eV/atom) and in static athermal calculations with 
  hybrid DFT (with dispersion corrections; HSE06+D3).
Citation: https://doi.org/10.1021/acs.jpcc.3c05204

## Miscellaneous Metals (Ag, Sn, Ti)
#### Silver (Ag)
- `mp-124` is the true FCC `Fm-3m` ground-state structure of silver, though with EaH ~ 0.002 eV/atom in the 
  MP database.
- Hybrid (HSE06) DFT predicts the `mp-989737` `R-3m` phase as the lowest energy silver allotrope in static
  athermal calculations (EaH ~ 0.010 eV/atom in the MP database).
Citation: https://doi.org/10.48550/arXiv.2602.22024

#### Tin (Sn)
- `mp-117` is the true diamond-like (cubic) `Fd-3m` ground-state structure of tin at low temperatures 
  (T ~< 286 K) – a.k.a. 'grey tin' or '⍺-tin', correctly predicted as the lowest energy tin phase in the 
  MP database (EaH = 0 eV/atom) and in static athermal calculations with hybrid DFT (with dispersion 
  corrections; HSE06+D3).
- `mp-84` is 'white tin' or 'β-tin', the ground-state tin crystal structure at higher temperatures 
  (T ~> 286 K), with an energy above hull (EaH) of ~0.12 eV/atom in the MP database (with static athermal 
  GGA DFT calculations).
Citation: https://doi.org/10.1021/acs.jpcc.3c05204

#### Titanium (Ti)
- `mp-46` is the true hexagonal close-packed (HCP) `P6₃/mmc` ground-state structure of titanium (for 
  temperatures T ~< 1550 K), though with EaH ~ 0.015 eV/atom in the MP database.
- `mp-72` is the high-pressure hexagonal ɷ (omega) allotrope of titanium, with EaH = 0 eV/atom in the MP 
  database.
- `mp-73` is the high-temperature β (beta) allotrope of titanium, stable for temperatures T ~> 1550 K, with 
  EaH = 0.15 eV/atom in the MP database.
Citation: https://doi.org/10.1021/acs.jpcc.3c05204

## Compounds
### Oxides (TiO₂, SnO₂, WO₃)
#### Titanium Dioxide (TiO₂)
- `mp-2657` is the true rutile (tetragonal) `P4₂/mnm` ground-state structure of titanium dioxide, though
  with EaH ~ 0.04 eV/atom in the MP database.
- `mp-390` is the anatase (tetragonal) `I4₁/amd` polymorph, being the lowest energy titanium dioxide phase 
  on the MP database (EaH = 0 eV/atom), and typically predicted as the lowest energy polymorph with hybrid DFT.
- `mp-1840` is the brookite (orthorhombic) `Pbca` polymorph, being the 2nd-lowest energy titanium dioxide 
  phase on the MP database (EaH ~ 0.003 eV/atom).
Citation: https://doi.org/10.1021/acs.jpcc.3c05204

There are many (other) low energy polymorphs of titanium dioxide on the MP database, but typically only 
these phases are relevant for competing phase calculations and chemical potentials.

#### Tin Dioxide (SnO₂)
- `mp-856` is the true rutile (tetragonal) `P4₂/mnm` ground-state structure of tin dioxide, correctly 
  predicted as the lowest energy tin dioxide phase in the MP database (EaH = 0 eV/atom), and with static hybrid DFT calculations.
Citation: https://doi.org/10.1021/acs.jpcc.3c05204

There are many (other) low energy polymorphs of tin dioxide on the MP database, but typically only this 
phase is relevant for competing phase calculations and chemical potentials.

#### Tungsten Tri-Oxide (WO₃)
WO₃ is a particularly difficult case, having many many low-energy polymorphs listed on the MP database, 
along with a number of 
[known phase transitions experimentally](https://pubs.acs.org/doi/10.1021/acs.jpclett.3c01546).
Depending on the use case (and if finite-temperature effects are being considered), one may want to prune 
the calculation set for this composition based on expected (relevant) ground-state phases.



Know of other cases with many low-energy polymorphs on the MP database, but known ground-state phases
Please submit a pull request to add to this list!
