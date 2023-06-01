[![Build status](https://github.com/SMTG-UCL/doped/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/SMTG-UCL/doped/actions)
[![PyPI](https://img.shields.io/pypi/v/doped)](https://pypi.org/project/doped)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/doped.svg)](https://anaconda.org/conda-forge/doped)
[![Downloads](https://img.shields.io/pypi/dm/doped)](https://shakenbreak.readthedocs.io/en/latest/)

# **D**efect **O**riented **P**ython **E**nvironment **D**istribution (`doped`)
`doped` is a python package for managing solid-state defect calculations, with functionality to 
generate defect structures and relevant competing phases (for chemical potentials), interface with 
[`ShakeNBreak`](https://shakenbreak.readthedocs.io) for 
[defect structure-searching](https://www.nature.com/articles/s41524-023-00973-1), write VASP input files for defect 
supercell calculations, and automatically parse and analyse the results.

Example Jupyter notebooks (the `.ipynb` files) are provided in [examples](examples) to show the code functionality and 
usage.

### Example Outputs:
Chemical potential/stability region plots and defect formation energy (a.k.a. transition level) diagrams:

<img src="https://raw.githubusercontent.com/SMTG-UCL/doped/master/files/doped_chempot_plotting.png" width="420">   &nbsp;&nbsp;  <img src="https://raw.githubusercontent.com/SMTG-UCL/doped/master/files/doped_TLD_plot.png" width="390">

## Requirements
`doped` requires `pymatgen>=2022.8.23` and its dependencies.

## Installation
1. Create virtual environment and install: 
```bash
pip install doped  # install doped and dependencies, can also  
pip install --force --no-cache-dir numpy==1.23 # install numpy after doped to avoid binary incompatibility
```

Alternatively if desired, `doped` can also be installed from `conda` with:

```bash
  conda install -c conda-forge doped
```

If you want to use the [example files](examples), 
you should clone the repository and install with `pip install -e .` from the `doped` directory, but still make sure to `pip install numpy --upgrade`.

2. (If not set) Set the VASP pseudopotential directory and your Materials Project API key in `$HOME/.pmgrc.yaml` 
(`pymatgen` config file) as follows:
```bash
  PMG_VASP_PSP_DIR: <Path to VASP pseudopotential top directory>
  PMG_MAPI_KEY: <Your MP API key obtained from https://legacy.materialsproject.org/open>
```
Within your `VASP pseudopotential top directory`, you should have a folder named 
`POT_GGA_PAW_PBE`/`potpaw_PBE.54`/`POT_GGA_PAW_PBE_54` which contains `POTCAR.X(.gz)` files, generated using `pmg config`.

If you have not previously setup your `POTCAR` directory in this way with `pymatgen`, then follow these steps:
```bash
mkdir temp_potcars  # make a top folder to store the unzipped POTCARs
mkdir temp_potcars/POT_PAW_GGA_PBE  # make a subfolder to store the unzipped POTCARs
mv potpaw_PBE.54.tar.gz temp_potcars/POT_PAW_GGA_PBE  # copy in your zipped VASP POTCAR source
cd temp_potcars/POT_PAW_GGA_PBE
tar -xzf potpaw_PBE.54.tar.gz  # unzip your VASP POTCAR source
cd ../..  # return to the top folder
pmg config -p temp_potcars psp_resources  # configure the psp_resources pymatgen POTCAR directory
pmg config --add PMG_VASP_PSP_DIR "${PWD}/psp_resources"  # add the POTCAR directory to pymatgen's config file (`$HOME/.pmgrc.yaml`)
rm -r temp_potcars  # remove the temporary POTCAR directory
```
If this has been successful, you should be able to run `pmg potcar -s Na_pv`, and `grep PBE POTCAR` should show 
`PAW_PBE Na_pv {date}` (you can ignore any `pymatgen` warnings about recognising the `POTCAR`). 

If it does not work check that the `PMG_DEFAULT_FUNCTIONAL` is set to whatever your functionals are (e.g. `PBE` or `PBE_54`)

This is necessary to generate `POTCAR` input files, and auto-determine `INCAR` settings such as `NELECT` for charged 
defects.

The Materials Project API key is required for determining the necessary competing phases to calculate in order to 
determine the chemical potential limits (required for defect formation energies). This should correspond to the legacy 
MP API, with your unique key available at: https://legacy.materialsproject.org/open.


## `ShakeNBreak`
As shown in the example notebook, it is highly recommended to use the [`ShakeNBreak`](https://shakenbreak.readthedocs.io/en/latest/) approach when calculating point defects in solids, to ensure you have identified the groundstate structures of your defects. As detailed in the [theory paper](https://arxiv.org/abs/2207.09862), skipping this step can result in drastically incorrect formation energies, transition levels, carrier capture (basically any property associated with defects). This approach is followed in the [doped example notebook](https://github.com/SMTG-UCL/doped/blob/master/dope_Example_Notebook.ipynb), with a more in-depth explanation and tutorial given on the [ShakeNBreak](https://shakenbreak.readthedocs.io/en/latest/) website.

Summary GIF:
![ShakeNBreak Summary](files/SnB_Supercell_Schematic_PES_2sec_Compressed.gif)

`SnB` CLI Usage:
![ShakeNBreak CLI](files/SnB_CLI.gif)


### Developer Installation

1. Download the `doped` source code using the command:
```bash
  git clone https://github.com/SMTG-UCL/doped
```
2. Navigate to root directory:
```bash
  cd doped
```
3. Install the code, using the command:
```bash
  pip install -e .
```

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
[`pymatgen-analysis-defects`](https://github.com/materialsproject/pymatgen-analysis-defects) package.

The colour scheme for defect formation energy plots was originally templated from 
[AIDE](https://github.com/SMTG-UCL/aide) (#neverforget), developed by the dynamic duo 
[Adam Jackson](https://github.com/ajjackson) and [Alex Ganose](https://github.com/utf).

