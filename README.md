[![Build status](https://github.com/SMTG-UCL/doped/actions/workflows/test.yml/badge.svg)](https://github.com/SMTG-UCL/doped/actions)
[![Documentation Status](https://readthedocs.org/projects/doped/badge/?version=latest&style=flat)](https://doped.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/doped)](https://pypi.org/project/doped)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/doped.svg)](https://anaconda.org/conda-forge/doped)
[![Downloads](https://img.shields.io/pypi/dm/doped)](https://pypi.org/project/doped)

<a href="https://doped.readthedocs.io/en/latest/"><img align="right" width="275" src="https://raw.githubusercontent.com/SMTG-UCL/doped/master/docs/doped_v2_logo.png"></a>`doped` is a python package for
managing solid-state defect calculations, with functionality to
generate defect structures and relevant competing phases (for chemical potentials), interface with
[`ShakeNBreak`](https://shakenbreak.readthedocs.io) for
[defect structure-searching](https://www.nature.com/articles/s41524-023-00973-1), write VASP input files for defect
supercell calculations, and automatically parse and analyse the results.

Tutorials showing the code functionality and usage are provided on the [docs](https://doped.readthedocs.io/en/latest/) site.

### Example Outputs:
Chemical potential/stability region plots and defect formation energy (a.k.a. transition level) diagrams:

<a href="https://doped.readthedocs.io/en/latest/dope_chemical_potentials.html#analysing-and-visualising-the-chemical-potential-limits"><img align="left" width="365" src="https://raw.githubusercontent.com/SMTG-UCL/doped/master/docs/doped_chempot_plotting.png"></a> <a href="https://doped.readthedocs.io/en/latest/dope_parsing_example.html#defect-formation-energy-transition-level-diagrams"><img align="right" width="385" src="https://raw.githubusercontent.com/SMTG-UCL/doped/master/docs/doped_TLD_plot.png"></a>
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
As shown in the example notebook, it is highly recommended to use the [`ShakeNBreak`](https://shakenbreak.readthedocs.io/en/latest/) approach when calculating point defects in solids, to ensure you have identified the groundstate structures of your defects. As detailed in the [theory paper](https://arxiv.org/abs/2207.09862), skipping this step can result in drastically incorrect formation energies, transition levels, carrier capture (basically any property associated with defects). This approach is followed in the [doped example notebook](https://github.com/SMTG-UCL/doped/blob/master/dope_workflow_example.ipynb), with a more in-depth explanation and tutorial given on the [ShakeNBreak](https://shakenbreak.readthedocs.io/en/latest/) website.

Summary GIF:
![ShakeNBreak Summary](https://raw.githubusercontent.com/SMTG-UCL/ShakeNBreak/main/docs/SnB_Supercell_Schematic_PES_2sec_Compressed.gif)

`SnB` CLI Usage:
![ShakeNBreak CLI](https://raw.githubusercontent.com/SMTG-UCL/ShakeNBreak/main/docs/SnB_CLI.gif)


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

## Studies using `doped` (so far)

- A. T. J. Nicolson et al. [_Journal of Materials Chemistry A_](https://doi.org/10.1039/D3TA02429F) 2023
- Y. W. Woo, Z. Li, Y-K. Jung, J-S. Park, A. Walsh [_ACS Energy Letters_](https://doi.org/10.1021/acsenergylett.2c02306) 2023
- P. A. Hyde et al. [_Inorganic Chemistry_](https://doi.org/10.1021/acs.inorgchem.3c01510) 2023
- J. Willis, K. B. Spooner, D. O. Scanlon. [_ChemRxiv_](https://chemrxiv.org/engage/chemrxiv/article-details/64c29140ce23211b20a787bb) 2023
- X. Wang et al. [_arXiv_](https://arxiv.org/abs/2302.04901) 2023
- J. Cen et al. [_Journal of Materials Chemistry A_](https://doi.org/10.1039/D3TA00532A) 2023
- J. Willis & R. Claes et al. [_ChemRxiv_](https://doi.org/10.26434/chemrxiv-2023-lttnf) 2023
- I. Mosquera-Lois & S. R. Kavanagh, A. Walsh, D. O. Scanlon [_npj Computational Materials_](https://www.nature.com/articles/s41524-023-00973-1) 2023
- Y. T. Huang & S. R. Kavanagh et al. [_Nature Communications_](https://www.nature.com/articles/s41467-022-32669-3) 2022
- S. R. Kavanagh, D. O. Scanlon, A. Walsh, C. Freysoldt [_Faraday Discussions_](https://doi.org/10.1039/D2FD00043A) 2022
- S. R. Kavanagh, D. O. Scanlon, A. Walsh [_ACS Energy Letters_](https://pubs.acs.org/doi/full/10.1021/acsenergylett.1c00380) 2021
- C. J. Krajewska et al. [_Chemical Science_](https://doi.org/10.1039/D1SC03775G) 2021
