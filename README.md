[![Build status](https://github.com/SMTG-Bham/doped/actions/workflows/test.yml/badge.svg)](https://github.com/SMTG-Bham/doped/actions)
[![Documentation Status](https://readthedocs.org/projects/doped/badge/?version=latest&style=flat)](https://doped.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/doped)](https://pypi.org/project/doped)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/doped.svg)](https://anaconda.org/conda-forge/doped)
[![Downloads](https://img.shields.io/pypi/dm/doped)](https://pypi.org/project/doped)

<a href="https://doped.readthedocs.io/en/latest/"><img align="right" width="275" src="https://raw.githubusercontent.com/SMTG-Bham/doped/main/docs/doped_v2_logo.png" alt="Schematic of a doped (defect-containing) crystal, inspired by the biological analogy to (semiconductor) doping." title="Schematic of a doped (defect-containing) crystal, inspired by the biological analogy to (semiconductor) doping."></a>`doped` is a python package for
managing solid-state defect calculations, with functionality to
generate defect structures and relevant competing phases (for chemical potentials), interface with
[`ShakeNBreak`](https://shakenbreak.readthedocs.io) for
[defect structure-searching](https://www.nature.com/articles/s41524-023-00973-1), write VASP input files for defect
supercell calculations, and automatically parse and analyse the results.

Tutorials showing the code functionality and usage are provided on the [docs](https://doped.readthedocs.io/en/latest/) site.

### Example Outputs:
Chemical potential/stability region plots and defect formation energy (a.k.a. transition level) diagrams:

<a href="https://doped.readthedocs.io/en/latest/chemical_potentials_tutorial.html#analysing-and-visualising-the-chemical-potential-limits"><img align="left" width="365" src="https://raw.githubusercontent.com/SMTG-Bham/doped/main/docs/doped_chempot_plotting.png"></a> <a href="https://doped.readthedocs.io/en/latest/dope_parsing_example.html#defect-formation-energy-transition-level-diagrams"><img align="right" width="385" src="https://raw.githubusercontent.com/SMTG-Bham/doped/main/docs/doped_TLD_plot.png"></a>
<br><br><br><br><br><br><br><br><br><br><br>


## Installation
```bash
pip install doped  # install doped and dependencies
```

Alternatively if desired, `doped` can also be installed from `conda` with:

```bash
  conda install -c conda-forge doped
```

If you haven't done so already, you will need to set up your VASP `POTCAR` files and `Materials Project` API with `pymatgen` using the `.pmgrc.yaml` file, in order for `doped` to automatically generate VASP input files for defect calculations and determine competing phases for chemical potentials.
See the docs [Installation](https://doped.readthedocs.io/en/latest/Installation.html) page for details on this.



## `ShakeNBreak`
As shown in the example notebook, it is highly recommended to use the [`ShakeNBreak`](https://shakenbreak.readthedocs.io/en/latest/) approach when calculating point defects in solids, to ensure you have identified the groundstate structures of your defects. As detailed in the [theory paper](https://doi.org/10.1038/s41524-023-00973-1), skipping this step can result in drastically incorrect formation energies, transition levels, carrier capture (basically any property associated with defects). This approach is followed in the [doped example notebook](https://github.com/SMTG-Bham/doped/blob/main/dope_workflow_example.ipynb), with a more in-depth explanation and tutorial given on the [ShakeNBreak](https://shakenbreak.readthedocs.io/en/latest/) website.

Summary GIF:
![ShakeNBreak Summary](https://raw.githubusercontent.com/SMTG-Bham/ShakeNBreak/main/docs/SnB_Supercell_Schematic_PES_2sec_Compressed.gif)

`SnB` CLI Usage:
![ShakeNBreak CLI](https://raw.githubusercontent.com/SMTG-Bham/ShakeNBreak/main/docs/SnB_CLI.gif)


## Acknowledgments
`doped` (née `DefectsWithTheBoys` #iykyk) has benefitted from feedback from many users, in particular
members of the [Scanlon](http://davidscanlon.com/) and [Walsh](https://wmd-group.github.io/) research groups who have used / are using it in their work. Direct contributors are listed in the `Contributors` sidebar above; including Seán Kavanagh, Bonan Zhu, Katarina Brlec, Adair Nicolson,
Sabrine Hachmioune and Savya Aggarwal.

Code to efficiently identify defect species from input supercell structures was contributed by Dr
[Alex Ganose](https://github.com/utf), and the colour scheme for defect formation energy plots was originally templated from
the `aide` package, developed by the dynamic duo [Adam Jackson](https://github.com/ajjackson) and [Alex Ganose](https://github.com/utf).

The [docs](https://readthedocs.io) website setup was templated from the `ShakeNBreak` docs set up by [Irea Mosquera-Lois](https://scholar.google.com/citations?user=oIMzt0cAAAAJ&hl=en) 🙌

`doped` was originally based on the excellent
[PyCDT](https://www.sciencedirect.com/science/article/pii/S0010465518300079) (no longer maintained), but transformed
and morphed over time as more and more functionality was added. After breaking changes in `pymatgen`, the package was
entirely refactored and rewritten, to work with the new
`pymatgen-analysis-defects` package.

## Studies using `doped`, so far

- X. Wang et al. **_Upper efficiency limit of Sb$\_2$Se$\_3$ solar cells_** [_arXiv_](https://arxiv.org/abs/2402.04434) 2024
- I. Mosquera-Lois et al. **_Machine-learning structural reconstructions for accelerated point defect calculations_** [_arXiv_](https://doi.org/10.48550/arXiv.2401.12127) 2024
- W. Dou et al. **_Giant Band Degeneracy via Orbital Engineering Enhances Thermoelectric Performance from Sb$_2$Si$_2$Te$_6$ to Sc$_2$Si$_2$Te$_6$_** [_ChemRxiv_](https://doi.org/10.26434/chemrxiv-2024-hm6vh) 2024
- K. Li et al. **_Computational Prediction of an Antimony-based n-type Transparent Conducting Oxide: F-doped Sb$_2$O$_5$_** [_ChemRxiv_](https://chemrxiv.org/engage/chemrxiv/article-details/65846b8366c1381729bc5f23) 2023
- X. Wang et al. **_Four-electron negative-U vacancy defects in antimony selenide_** [_Physical Review B_](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.108.134102) 2023
- Y. Kumagai et al. **_Alkali Mono-Pnictides: A New Class of Photovoltaic Materials by Element Mutation_** [_PRX Energy_](http://dx.doi.org/10.1103/PRXEnergy.2.043002) 2023
- S. M. Liga & S. R. Kavanagh, A. Walsh, D. O. Scanlon, G. Konstantatos **_Mixed-Cation Vacancy-Ordered Perovskites (Cs$_2$Ti$_{1–x}$Sn$_x$X$_6$; X = I or Br): Low-Temperature Miscibility, Additivity, and Tunable Stability_*** [_Journal of Physical Chemistry C_](https://doi.org/10.1021/acs.jpcc.3c05204) 2023
- A. T. J. Nicolson et al. **_Cu$_2$SiSe$_3$ as a promising solar absorber: harnessing cation dissimilarity to avoid killer antisites_** [_Journal of Materials Chemistry A_](https://doi.org/10.1039/D3TA02429F) 2023
- Y. W. Woo, Z. Li, Y-K. Jung, J-S. Park, A. Walsh **_Inhomogeneous Defect Distribution in Mixed-Polytype Metal Halide Perovskites_** [_ACS Energy Letters_](https://doi.org/10.1021/acsenergylett.2c02306) 2023
- P. A. Hyde et al. **_Lithium Intercalation into the Excitonic Insulator Candidate Ta$_2$NiSe$_5$_** [_Inorganic Chemistry_](https://doi.org/10.1021/acs.inorgchem.3c01510) 2023
- J. Willis, K. B. Spooner, D. O. Scanlon **_On the possibility of p-type doping in barium stannate_** [_Applied Physics Letters_](https://doi.org/10.1063/5.0170552) 2023
- J. Cen et al. **_Cation disorder dominates the defect chemistry of high-voltage LiMn$_{1.5}$Ni$_{0.5}$O$_4$ (LMNO) spinel cathodes_** [_Journal of Materials Chemistry A_](https://doi.org/10.1039/D3TA00532A) 2023
- J. Willis & R. Claes et al. **_Limits to Hole Mobility and Doping in Copper Iodide_** [_Chem Mater_](https://doi.org/10.1021/acs.chemmater.3c01628) 2023
- I. Mosquera-Lois & S. R. Kavanagh, A. Walsh, D. O. Scanlon **_Identifying the ground state structures of point defects in solids_** [_npj Computational Materials_](https://www.nature.com/articles/s41524-023-00973-1) 2023
- Y. T. Huang & S. R. Kavanagh et al. **_Strong absorption and ultrafast localisation in NaBiS$\_2$ nanocrystals with slow charge-carrier recombination_** [_Nature Communications_](https://www.nature.com/articles/s41467-022-32669-3) 2022
- S. R. Kavanagh, D. O. Scanlon, A. Walsh, C. Freysoldt **_Impact of metastable defect structures on carrier recombination in solar cells_** [_Faraday Discussions_](https://doi.org/10.1039/D2FD00043A) 2022
- Y-S. Choi et al. **_Intrinsic Defects and Their Role in the Phase Transition of Na-Ion Anode Na$_2$Ti$_3$O$_7$** [_ACS Appl. Energy Mater._](https://doi.org/10.1021/acsaem.2c03466) 2022
- S. R. Kavanagh, D. O. Scanlon, A. Walsh **_Rapid Recombination by Cadmium Vacancies in CdTe_** [_ACS Energy Letters_](https://pubs.acs.org/doi/full/10.1021/acsenergylett.1c00380) 2021
- C. J. Krajewska et al. **_Enhanced visible light absorption in layered Cs$_3$Bi_2$Br_9$ through mixed-valence Sn(II)/Sn(IV) doping_** [_Chemical Science_](https://doi.org/10.1039/D1SC03775G) 2021
