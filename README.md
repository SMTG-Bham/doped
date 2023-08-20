[![Build status](https://github.com/SMTG-UCL/doped/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/SMTG-UCL/doped/actions)
[![Documentation Status](https://readthedocs.org/projects/doped/badge/?version=latest&style=flat)](https://doped.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/doped)](https://pypi.org/project/doped)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/doped.svg)](https://anaconda.org/conda-forge/doped)
[![Downloads](https://img.shields.io/pypi/dm/doped)](https://pypi.org/project/doped)

<img align="right" width="275" src="docs/doped_v2_logo.png">`doped` is a python package for
managing solid-state defect calculations, with functionality to
generate defect structures and relevant competing phases (for chemical potentials), interface with
[`ShakeNBreak`](https://shakenbreak.readthedocs.io) for
[defect structure-searching](https://www.nature.com/articles/s41524-023-00973-1), write VASP input files for defect
supercell calculations, and automatically parse and analyse the results.

Tutorials showing the code functionality and usage are provided on the [docs](https://doped.readthedocs.io/en/latest/) site.

### Example Outputs:
Chemical potential/stability region plots and defect formation energy (a.k.a. transition level) diagrams:

<img align="left" src="docs/doped_chempot_plotting.png" width="365"> <img src="docs/doped_TLD_plot.png" width="385" align="right">

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
As shown in the example notebook, it is highly recommended to use the [`ShakeNBreak`](https://shakenbreak.readthedocs.io/en/latest/) approach when calculating point defects in solids, to ensure you have identified the groundstate structures of your defects. As detailed in the [theory paper](https://arxiv.org/abs/2207.09862), skipping this step can result in drastically incorrect formation energies, transition levels, carrier capture (basically any property associated with defects). This approach is followed in the [doped example notebook](https://github.com/SMTG-UCL/doped/blob/master/dope_Example_Notebook.ipynb), with a more in-depth explanation and tutorial given on the [ShakeNBreak](https://shakenbreak.readthedocs.io/en/latest/) website.

Summary GIF:
![ShakeNBreak Summary](docs/SnB_Supercell_Schematic_PES_2sec_Compressed.gif)

`SnB` CLI Usage:
![ShakeNBreak CLI](docs/SnB_CLI.gif)


## Acknowledgments
`doped` (née `DefectsWithTheBoys` #gonebutnotforgotten) has benefitted from feedback from many users, in particular
members of the Walsh and Scanlon research groups who have used / are using it in their work. Direct contributors are
listed in the `Contributors` sidebar above; including Seán Kavanagh, Bonan Zhu, Katarina Brlec, Adair Nicolson,
Sabrine Hachmioune and Savya Aggarwal.

Code to efficiently identify defect species from input supercell structures was contributed by Dr Alex Ganose.

`doped` was originally based on the excellent
[PyCDT](https://www.sciencedirect.com/science/article/pii/S0010465518300079) (no longer maintained), but transformed
and morphed over time as more and more functionality was added. After breaking changes in `pymatgen`, the package was
entirely refactored and rewritten, to work with the new
`pymatgen-analysis-defects` package.

The colour scheme for defect formation energy plots was originally templated from `aide` (#neverforget), developed by the dynamic duo
[Adam Jackson](https://github.com/ajjackson) and [Alex Ganose](https://github.com/utf).
