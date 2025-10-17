[![Build status](https://github.com/SMTG-Bham/doped/actions/workflows/test.yml/badge.svg)](https://github.com/SMTG-Bham/doped/actions)
[![Documentation Status](https://readthedocs.org/projects/doped/badge/?version=latest&style=flat)](https://doped.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/doped)](https://pypi.org/project/doped)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/doped.svg)](https://anaconda.org/conda-forge/doped)
[![Downloads](https://img.shields.io/pypi/dm/doped)](https://pypi.org/project/doped)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06433/status.svg)](https://doi.org/10.21105/joss.06433)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SMTG-Bham/doped)

<a href="https://doped.readthedocs.io/en/latest/"><img align="right" width="150" src="https://raw.githubusercontent.com/SMTG-Bham/doped/main/docs/Images/doped_v2_logo.png" alt="Schematic of a doped (defect-containing) crystal, inspired by the biological analogy to (semiconductor) doping." title="Schematic of a doped (defect-containing) crystal, inspired by the biological analogy to (semiconductor) doping."></a>`doped` is a Python software for the generation, pre-/post-processing and analysis of defect supercell calculations, implementing the defect simulation workflow in an efficient, reproducible, user-friendly yet powerful and fully-customisable manner.

Tutorials showing the code functionality and usage are provided on the [docs](https://doped.readthedocs.io/en/latest/) site, and an overview of the key advances of the package is given in the [JOSS paper](https://doi.org/10.21105/joss.06433).

![https://joss.theoj.org/papers/10.21105/joss.06433](docs/JOSS/doped_JOSS_workflow_figure.png)

## Key Features
All features and functionality are fully-customisable:
- **Supercell Generation**: Generate an optimal supercell, maximising periodic image separation for the minimum number of atoms (computational cost).
- **Defect Generation**: Generate defect supercells and guess likely charge states based on chemical intuition.
- **Calculation I/O**: Automatically write inputs and parse calculations (`VASP` & other DFT/force-field codes).
- **Chemical Potentials**: Determine relevant competing phases for chemical potential limits, with automated calculation setup, parsing and analysis.
- **Defect Analysis**: Automatically parse calculation outputs to compute defect formation energies, finite-size corrections (FNV & eFNV), symmetries, degeneracies, transition levels, etc.
- **Thermodynamic Analysis**: Compute (non-)equilibrium Fermi levels, defect/carrier concentrations etc. as functions of annealing/cooling temperature, chemical potentials etc.
- **Plotting**: Generate publication-quality plots of defect formation energies, chemical potential limits, defect/carrier concentrations, Fermi levels, charge corrections, etc.
- **`Python` Interface**: Fully-customisable, modular `Python` API. Plug-and-play w/[`ShakeNBreak`](https://shakenbreak.readthedocs.io) – [defect structure-searching](https://www.nature.com/articles/s41524-023-00973-1), [`easyunfold`](https://smtg-bham.github.io/easyunfold/) – band unfolding, [`CarrierCapture.jl`](https://github.com/WMD-group/CarrierCapture.jl)/[`nonrad`](https://nonrad.readthedocs.io/en/latest/) – non-radiative recombination etc.
- Reproducibility, tabulation, automated compatibility/sanity checking, strain/displacement analysis, shallow defect analysis, high-throughput compatibility, Wyckoff analysis...

### Performance and Example Outputs
![https://joss.theoj.org/papers/10.21105/joss.06433](docs/JOSS/doped_JOSS_figure.png)
**(a)** Optimal supercell generation comparison. **(b)** Charge state estimation comparison. Example **(c)** Kumagai-Oba (eFNV) finite-size correction plot, **(d)** defect formation energy diagram, **(e)** chemical potential / stability region, **(f)** Fermi level vs. annealing temperature, **(g)** defect/carrier concentrations vs. annealing temperature and **(h)** Fermi level / carrier concentration heatmap plots from `doped`. Automated plots of **(i,j)** single-particle eigenvalues and **(k)** site
displacements from DFT supercell calculations. See the [JOSS paper](https://doi.org/10.21105/joss.06433) for more details.

## Installation
```bash
pip install doped  # install doped and dependencies
```

If you want to install the latest development version of `doped`, this can be done following the
[Developer Installation](https://doped.readthedocs.io/en/latest/Installation.html#developer-installation) instructions,
or with:
```bash
pip install https://github.com/SMTG-Bham/doped/archive/develop.zip
```

Alternatively if desired, `doped` can also be installed from `conda` with:

```bash
  conda install -c conda-forge doped
  pip install pydefect  # pydefect not available on conda, so needs to be installed with pip or otherwise, if using the eFNV correction
```

See the [Installation docs](https://doped.readthedocs.io/en/latest/Installation.html) if you encounter any issues (e.g. known issue with `phonopy` `CMake` build).

If you haven't done so already, you will need to set up your VASP `POTCAR` files and `Materials Project` API with `pymatgen` using the `.pmgrc.yaml` file, in order for `doped` to automatically generate VASP input files for defect calculations and determine competing phases for chemical potentials.
See the docs [Installation](https://doped.readthedocs.io/en/latest/Installation.html) page for details on this.

## Citation
If you use `doped` in your research, please cite:
- S. R. Kavanagh et al. [doped: Python toolkit for robust and repeatable charged defect supercell calculations](https://doi.org/10.21105/joss.06433). _Journal of Open Source Software_ 9 (96), 6433, **2024**

## Literature
The following literature contain useful discussions of various aspects of defect calculations:
- [Quick-Start Guide on Defect Calculations – Kim et al.](https://doi.org/10.1088/2515-7655/aba081)
- [Large Review on Defect Calculations – Freysoldt et al.](https://doi.org/10.1103/RevModPhys.86.253)
- [Guide to Understanding Formation Energy / Transition Level Diagrams – Gorai](https://3d-materials-lab.gitbook.io/3dmaterialslab-tutorials/defects/interpreting-defect-and-energy-level-diagrams)
- [Defect Structure Searching – Mosquera-Lois et al.](https://doi.org/10.1038/s41524-023-00973-1)
- [Free Energies of Defects – Mosquera-Lois et al.](https://doi.org/10.1039/D3CS00432E)
- [DeepWiki Code Overview & Workflow](https://deepwiki.com/SMTG-Bham/doped/1-overview)
<!--- TODO: Squires perspective when ready --->

## `ShakeNBreak`
As shown in the `doped` tutorials, it is highly recommended to use the [`ShakeNBreak`](https://shakenbreak.readthedocs.io/en/latest/) approach when calculating point defects in solids, to ensure you have identified the groundstate structures of your defects. As detailed in the [theory paper](https://doi.org/10.1038/s41524-023-00973-1), skipping this step can result in drastically incorrect formation energies, transition levels, carrier capture (basically any property associated with defects). This approach is followed in the [doped defect generation tutorial](https://doped.readthedocs.io/en/latest/generation_tutorial.html), with a more in-depth explanation and tutorial given on the [ShakeNBreak](https://shakenbreak.readthedocs.io/en/latest/) website.

![ShakeNBreak Summary](https://raw.githubusercontent.com/SMTG-Bham/ShakeNBreak/main/docs/SnB_Supercell_Schematic_PES_2sec_Compressed.gif)

## Studies using `doped`, so far

- P. Russell et al. **_Computational prediction of Y-doped Cd<sub>2</sub>Sb<sub>2</sub>O<sub>7</sub> as a competitive Sb-based n-type Transparent Conducting Oxide_** [_ChemRxiv_](https://doi.org/10.26434/chemrxiv-2025-c1r3l) 2025
- A. Lannerd et al. **_Thermodynamics of proton insertion across the perovskite-brownmillerite transition in La<sub>0.5</sub>Sr<sub>0.5</sub>CoO<sub>3-δ</sub>_** [_arXiv_](https://arxiv.org/abs/2510.05323) 2025
- V. Watson et al. **_Enhanced Oxide Ion Diffusion by Lanthanum Substitution in the Palmierite Sr<sub>3–3x</sub>La<sub>2x</sub>V<sub>2</sub>O<sub>8</sub> via Increased Tetrahedral Distortion and Cation Vacancies_** [_Chemistry of Materials_](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.5c01856) 2025
- P. Deng et al. **_Carbon Interstitial Diffusion in 3C-SiC: Role of Charge-State Transitions and Entropy_** [_Advanced Theory and Simulations_](https://doi.org/10.1002/adts.202501088) 2025
- P. Panagis Filippatos, T. J. P. Irons and K. Inzani **_Revisiting intrinsic spin defects in hexagonal boron nitride with r2SCAN_** [_arXiv_](http://dx.doi.org/10.48550/arXiv.2509.09002) 2025
- P. Panagis Filippatos, A. Chroneos and N. Kelaidis **_A first-principles investigation of halogen doped diamond and its application to quantum technologies_** [_Journal of Applied Physics_](https://doi.org/10.1063/5.0279139) 2025
- L. Tian et al. **_Optical Spectroscopic Determination of Photoexcited Small-Polaron Hopping in Transition Metal Oxide Photocatalysts_** [_ChemRxiv_](https://doi.org/10.26434/chemrxiv-2025-bhf0z) 2025
- M. A. Fadla et al. **_Tailoring the Electronic Properties of Monoclinic (In<sub>x</sub>Al<sub>1-x</sub>)<sub>2</sub>O<sub>3</sub> Alloys via Substitutional Donors and Acceptors_** [_arXiv_](https://arxiv.org/abs/2507.02805) 2025
- M. E. Turiansky et al. **_Machine Learning Phonon Spectra for
Fast and Accurate Optical Lineshapes of Defects_** [_arXiv_](https://arxiv.org/abs/2508.09113) 2025
- C. López et al. **_Chalcogen Vacancies Rule Charge Recombination in Pnictogen Chalcohalide Solar-Cell Absorbers_** [_ACS Energy Letters_](https://pubs.acs.org/doi/10.1021/acsenergylett.5c01267) 2025
- S. R. Kavanagh **_Identifying Split Vacancy Defects with Machine-Learned Foundation Models and Electrostatics_** [_JPhys Energy_](https://doi.org/10.1088/2515-7655/ade916) 2025
- K. Ogawa et al. **_Defect Tolerance via External Passivation in the Photocatalyst SrTiO<sub>3</sub>:Al_** [_Journal of the American Chemical Society_](https://doi.org/10.1021/jacs.5c07104) 2025
- M. S. Islam et al. **_Diffusion Characteristics of Ru and Oxygen Vacancies in Ta<sub>2</sub>O<sub>5</sub> for Resistive Random Access Memory Devices: A Density Functional Theory Investigation_** [_Advanced Electronic Materials_](https://doi.org/10.1002/aelm.202500128) 2025
- J. Tu et al. **_Giant switchable ferroelectric photovoltage in double-perovskite epitaxial films through chemical negative strain_** [_Science Advances_](https://doi.org/10.1126/sciadv.ads4925) 2025
- Y. Fu & H. Lohan et al. **_Factors Enabling Delocalized Charge-Carriers in Pnictogen-Based
Solar Absorbers: In-depth Investigation into CuSbSe<sub>2</sub>_** [_Nature Communications_](https://doi.org/10.1038/s41467-024-55254-2) 2025
- S. R. Kavanagh et al. **_Intrinsic point defect tolerance in selenium for indoor and tandem photovoltaics_** [_Energy & Environmental Science_](https://doi.org/10.1039/D4EE04647A) 2025
- N. T. Taylor et al. **_Instability of oxide perovskite surfaces induced by vacancy formation_** [_Physical Review B_](https://doi.org/10.1103/w683-tvc4) 2025
- J. Hu et al. **_Enabling ionic transport in Li<sub>3</sub>AlP<sub>2</sub> the roles of defects and disorder_** [_Journal of Materials Chemistry A_](https://doi.org/10.1039/D4TA04347B) 2025
- X. Jiang et al. **_Carrier lifetime killer in 4H-SiC: carrier capture path via carbon vacancies_** [_Journal of Materials Chemistry C_](https://doi.org/10.1039/D4TC04558K) 2025
- M. R. Khan et al. **_Interplay between intrinsic defects and optoelectronic properties of semi-Heusler gapped metals_** [_Physical Chemistry Chemical Physics_](https://doi.org/10.1039/D5CP00673B) 2025
- F. Wang et al. **_Investigation of hydrogen diffusion in zirconia under extreme conditions_** [_Physical Chemistry Chemical Physics_](https://doi.org/10.1039/D5CP00434A) 2025
- R. Chinnappan **_First-principles study of defect energetics and magnetic properties of Cr, Ru and Rh doped AlN_** [_Physica Scripta_](https://doi.org/10.1088/1402-4896/adca71) 2025
- R. Desai et al. **_Exploring the Defect Landscape and Dopability of Chalcogenide Perovskite BaZrS<sub>3</sub>_** [_Journal of Physical Chemistry C_](https://doi.org/10.1021/acs.jpcc.5c01597) 2025
- C. Kaewmeechai, J. Strand & A. Shluger **_Structure and Migration Mechanisms of Oxygen Interstitial Defects in β-Ga<sub>2</sub>O<sub>3</sub>_** [_Physica Status Solidi B_](https://onlinelibrary.wiley.com/doi/10.1002/pssb.202400652) 2025 <!-- though didn't cite properly SnB or doped code papers... -->
- W. Gierlotka et al. **_Thermodynamics of point defects in the AlSb phase and its influence on phase equilibrium_** [_Computational Materials Science_](https://doi.org/10.1016/j.commatsci.2025.113934) 2025 <!-- didn't cite SnB code paper... -->
- M. Biswas et al. **_Unified Graph-based Interatomic Potential for Perovskite Structure Optimization_** [_ChemRxiv_](https://doi.org/10.26434/chemrxiv-2025-g9sb9) 2025
- X. Wang et al. **_Sulfur Vacancies Limit the Open-circuit Voltage of Sb<sub>2</sub>S<sub>3</sub> Solar Cells_** [_ACS Energy Letters_](https://doi.org/10.1021/acsenergylett.4c02722) 2024
- A. Zhang et al. **_Optimizing the n-type carrier concentration of an InVO<sub>4</sub> photocatalyst by codoing with donors and intrinsic defects_** [_Physical Review Applied_](https://doi.org/10.1103/PhysRevApplied.22.044047) 2024
- M-L. Wang et al. **_Impact of sulfur doping on copper-substituted lead apatite_** [_Physical Review B_](https://doi.org/10.1103/PhysRevB.110.104109) 2024
- S. Quadir et al. **_Low-Temperature Synthesis of Stable CaZn<sub>2</sub>P<sub>2</sub> Zintl Phosphide Thin Films as Candidate Top Absorbers_** [_Advanced Energy Materials_](https://doi.org/10.1002/aenm.202402640) 2024
- M. Elgaml et al. **_Controlling the Superconductivity of Nb<sub>2</sub>Pd<sub>x</sub>S<sub>5</sub> via Reversible Li Intercalation_** [_Inorganic Chemistry_](https://pubs.acs.org/doi/full/10.1021/acs.inorgchem.3c03524) 2024
- Z. Yuan & G. Hautier **_First-principles study of defects and doping limits in CaO_** [_Applied Physics Letters_](https://doi.org/10.1063/5.0211707) 2024
- B. E. Murdock et al. **_Li-Site Defects Induce Formation of Li-Rich Impurity Phases: Implications for Charge Distribution and Performance of LiNi<sub>0.5-x</sub>M<sub>x</sub>Mn<sub>1.5</sub>O<sub>4</sub> Cathodes (M = Fe and Mg; x = 0.05–0.2)_** [_Advanced Materials_](https://doi.org/10.1002/adma.202400343) 2024
- A. G. Squires et al. **_Oxygen dimerization as a defect-driven process in bulk LiNiO2<sub>2</sub>_** [_ACS Energy Letters_](https://pubs.acs.org/doi/10.1021/acsenergylett.4c01307) 2024
- X. Wang et al. **_Upper efficiency limit of Sb<sub>2</sub>Se<sub>3</sub> solar cells_** [_Joule_](https://doi.org/10.1016/j.joule.2024.05.004) 2024
- I. Mosquera-Lois et al. **_Machine-learning structural reconstructions for accelerated point defect calculations_** [_npj Computational Materials_](https://doi.org/10.1038/s41524-024-01303-9) 2024
- W. Dou et al. **_Band Degeneracy and Anisotropy Enhances Thermoelectric Performance from Sb<sub>2</sub>Si<sub>2</sub>Te<sub>6</sub> to Sc<sub>2</sub>Si<sub>2</sub>Te<sub>6</sub>_** [_Journal of the American Chemical Society_](https://doi.org/10.1021/jacs.4c01838) 2024
- K. Li et al. **_Computational Prediction of an Antimony-based n-type Transparent Conducting Oxide: F-doped Sb<sub>2</sub>O<sub>5</sub>_** [_Chemistry of Materials_](https://doi.org/10.1021/acs.chemmater.3c03257) 2024
- S. Hachmioune et al. **_Exploring the Thermoelectric Potential of MgB<sub>4</sub>: Electronic Band Structure, Transport Properties, and Defect Chemistry_** [_Chemistry of Materials_](https://doi.org/10.1021/acs.chemmater.4c00584) 2024
- Y. Zeng et al. **_Role of carbon in α-Al2O3:C crystals investigated with first-principles calculations and experiment_** [_Ceramics International_](https://doi.org/10.1016/j.ceramint.2024.12.512) 2024
- X. Wang et al. **_Four-electron negative-U vacancy defects in antimony selenide_** [_Physical Review B_](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.108.134102) 2023
- Y. Kumagai et al. **_Alkali Mono-Pnictides: A New Class of Photovoltaic Materials by Element Mutation_** [_PRX Energy_](http://dx.doi.org/10.1103/PRXEnergy.2.043002) 2023
- S. M. Liga & S. R. Kavanagh, A. Walsh, D. O. Scanlon, G. Konstantatos **_Mixed-Cation Vacancy-Ordered Perovskites (Cs<sub>2</sub>Ti<sub>1–x</sub>Sn<sub>x</sub>X<sub>6</sub>; X = I or Br): Low-Temperature Miscibility, Additivity, and Tunable Stability_** [_Journal of Physical Chemistry C_](https://doi.org/10.1021/acs.jpcc.3c05204) 2023
- A. T. J. Nicolson et al. **_Cu<sub>2</sub>SiSe<sub>3</sub> as a promising solar absorber: harnessing cation dissimilarity to avoid killer antisites_** [_Journal of Materials Chemistry A_](https://doi.org/10.1039/D3TA02429F) 2023
- Y. W. Woo, Z. Li, Y-K. Jung, J-S. Park, A. Walsh **_Inhomogeneous Defect Distribution in Mixed-Polytype Metal Halide Perovskites_** [_ACS Energy Letters_](https://doi.org/10.1021/acsenergylett.2c02306) 2023
- P. A. Hyde et al. **_Lithium Intercalation into the Excitonic Insulator Candidate Ta<sub>2</sub>NiSe<sub>5</sub>_** [_Inorganic Chemistry_](https://doi.org/10.1021/acs.inorgchem.3c01510) 2023
- J. Willis, K. B. Spooner, D. O. Scanlon **_On the possibility of p-type doping in barium stannate_** [_Applied Physics Letters_](https://doi.org/10.1063/5.0170552) 2023
- J. Cen et al. **_Cation disorder dominates the defect chemistry of high-voltage LiMn<sub>1.5</sub>Ni<sub>0.5</sub>O<sub>4</sub> (LMNO) spinel cathodes_** [_Journal of Materials Chemistry A_](https://doi.org/10.1039/D3TA00532A) 2023
- J. Willis & R. Claes et al. **_Limits to Hole Mobility and Doping in Copper Iodide_** [_Chemistry of Materials_](https://doi.org/10.1021/acs.chemmater.3c01628) 2023
- I. Mosquera-Lois & S. R. Kavanagh, A. Walsh, D. O. Scanlon **_Identifying the ground state structures of point defects in solids_** [_npj Computational Materials_](https://www.nature.com/articles/s41524-023-00973-1) 2023
- Y. T. Huang & S. R. Kavanagh et al. **_Strong absorption and ultrafast localisation in NaBiS<sub>2</sub> nanocrystals with slow charge-carrier recombination_** [_Nature Communications_](https://www.nature.com/articles/s41467-022-32669-3) 2022
- S. R. Kavanagh, D. O. Scanlon, A. Walsh, C. Freysoldt **_Impact of metastable defect structures on carrier recombination in solar cells_** [_Faraday Discussions_](https://doi.org/10.1039/D2FD00043A) 2022
- Y-S. Choi et al. **_Intrinsic Defects and Their Role in the Phase Transition of Na-Ion Anode Na<sub>2</sub>Ti<sub>3</sub>O<sub>7</sub>_** [_ACS Applied Energy Materials_](https://doi.org/10.1021/acsaem.2c03466) 2022
- S. R. Kavanagh, D. O. Scanlon, A. Walsh **_Rapid Recombination by Cadmium Vacancies in CdTe_** [_ACS Energy Letters_](https://pubs.acs.org/doi/full/10.1021/acsenergylett.1c00380) 2021
- C. J. Krajewska et al. **_Enhanced visible light absorption in layered Cs<sub>3</sub>Bi<sub>2</sub>Br<sub>9</sub> through mixed-valence Sn(II)/Sn(IV) doping_** [_Chemical Science_](https://doi.org/10.1039/D1SC03775G) 2021

## Acknowledgments
`doped` (née `DefectsWithTheBoys`) has benefitted from feedback from many users, in particular
members of the [Scanlon](http://davidscanlon.com/) and [Walsh](https://wmd-group.github.io/) research groups who have used / are using it in their work. Direct contributors are listed in the `Contributors` sidebar above; including Seán Kavanagh, Alex Squires, Adair Nicolson, Irea Mosquera-Lois, Alex Ganose, Bonan Zhu, Katarina Brlec, Sabrine Hachmioune and Savya Aggarwal.

`doped` was originally based on the excellent `PyCDT` (no longer maintained), but transformed and morphed over time as more and more functionality was added. After breaking changes in `pymatgen`, the package was entirely refactored and rewritten, to work with the new
`pymatgen-analysis-defects` package.

Thanks to [Chaoqun Zhang](https://github.com/Warlocat) for uploading the [YouTube tutorials](https://youtu.be/FWz7nm9qoNg) with Chinese
subtitles to [Bilibili](https://www.bilibili.com/list/6073855/?sid=4603908)!
