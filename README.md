# DefectsWithTheBoys
This is a (mid-development) Python package for managing solid-state defect calculations,
geared toward VASP. Much of it is a modified version of the excellent [PyCDT](https://bitbucket.org/mbkumar/pycdt).  
See [this link](https://www.sciencedirect.com/science/article/pii/S0010465518300079) for the original PyCDT paper.


This code is still being customised, so in the spirit of efficiency 
and avoiding redundant work, I've just provided an example 
[Jupyter notebook](DWTB_Example_Notebook.ipynb)
of the code functionality and usage, 
so please look at that. (Better to open in Jupyter, after installing, rather than with GitHub preview).

If I reach a final product at some point 
(likely integrating things like [CPLAP](https://github.com/jbuckeridge/cplap), 
[SC-Fermi](https://github.com/jbuckeridge/sc-fermi), `AIDE` etc.),
 I'll make a detailed README then.
 And a more appropriate name, maybe...


## Requirements
DefectsWithTheBoys requires pymatgen (and its dependencies).

## Installation
1.  Download the DefectsWithTheBoys source code using the command:
```bash
  git clone https://github.com/kavanase/DefectsWithTheBoys
```
2.  Navigate to root directory:
```bash
  cd DefectsWithTheBoys
```
3.  Install the code, using the command:
```bash
  python setup.py install
```
This command tries to obtain the required packages and their dependencies and install them automatically.
Access to root may be needed if ``virtualenv`` is not used.

4.  (If not set) Set the VASP pseudopotential directory in `$HOME/.pmgrc.yaml` as follows::
```bash
  VASP_PSP_DIR: <Location of vasp pseudopotential top directory>
```

5.  (If not set) Set the Materials Project API key in `$HOME/.pmgrc.yaml` as follows::
```bash
  MAPI_KEY: <Your mapi key obtained from www.materialsproject.org>
```

## Word of Caution
There is quite possibly a couple of bugs in this code, as it is very much still experimental and in development.
