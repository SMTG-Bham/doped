# **D**efect **O**riented **P**ython **E**nvironment **D**istribution (`doped`)
This is a (mid-development) Python package for managing solid-state defect calculations,
geared toward VASP. Much of it is a modified version of the excellent [PyCDT](https://bitbucket.org/mbkumar/pycdt).  
See [this link](https://www.sciencedirect.com/science/article/pii/S0010465518300079) for the original PyCDT paper.

Defect formation energy plots are templated from [AIDE](https://github.com/SMTG-UCL/aide) and follow the aesthetics
philosopy of [sumo](https://smtg-ucl.github.io/sumo/), both developed by the dynamic duo Adam Jackson and Alex Ganose.

Example Jupyter notebooks (the `.ipynb` files) are provided in [examples](examples) to show the code functionality and usage.

## Requirements
`doped` requires pymatgen (and its dependencies).

## Installation
1. `doped` can be installed from `PyPI` with `pip install doped`. 

2. (If not set) Set the VASP pseudopotential directory and your Materials Project API key in `$HOME/.pmgrc.yaml` 
(`pymatgen` config file) as follows:
```bash
  PMG_VASP_PSP_DIR: <Path to VASP pseudopotential top directory>
  PMG_MAPI_KEY: <Your MP API key obtained from https://legacy.materialsproject.org/open>
```
Within your `VASP pseudopotential top directory`, you should have a folder named 
`POT_GGA_PAW_PBE`/`potpaw_PBE.54`/`POT_GGA_PAW_PBE_54` which contains `POTCAR.X(.gz)` files, generated using `pmg config`.

If you have not previously setup your `POTCAR` directory in this way with `pymatgen`, then follow these steps:
    1. Download and unzip your VASP pseudopotentials:
    ```bash
    mkdir potpaw_PBE  # make a folder to store the unzipped POTCARs
    cp potpaw_PBE.54.tar.gz potpaw_PBE  # copy your zipped VASP POTCAR source to this folder
    cd potpaw_PBE
    tar -xf potpaw_PBE.54.tar.gz  # unzip your VASP POTCAR source
    pmg config -p . POT_PAW_GGA_PBE
    pmg config --add PMG_VASP_PSP_DIR ./POT_PAW_GGA_PBE
    ```
    If this has been successful, you should be able to run `pmg potcar -s Na_pv`, and `grep PBE POTCAR` should show 
    `PAW_PBE Na_pv {date}` 

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

Note that while [ShakeNBreak](https://shakenbreak.readthedocs.io/en/latest/) is built to be compatible with the latest `pymatgen` version, the defects corrections code has been removed from the current `pymatgen` version, so when installing [ShakeNBreak](https://shakenbreak.readthedocs.io/en/latest/) the `2022.11.1` version should be used, with: `pip install shakenbreak==2022.11.1`.


### Developer Installation

1. Download the `doped` source code using the command:
```bash
  git clone https://github.com/SMTG-UCL/doped
```
2.  Navigate to root directory:
```bash
  cd doped
```
3.  Install the code, using the command:
```bash
  pip install -e .
```
This command tries to obtain the required packages and their dependencies and install them automatically.
Access to root may be needed if ``virtualenv`` is not used.


## Word of Caution
There is quite possibly a couple of bugs in this code, as it is very much still experimental and in development.
If you find any, please let us know!
