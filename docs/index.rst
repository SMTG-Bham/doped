.. image:: https://github.com/SMTG-UCL/doped/actions/workflows/build_and_test.yml/badge.svg
   :target: https://github.com/SMTG-UCL/doped/actions
.. image:: https://img.shields.io/pypi/v/doped
   :target: https://pypi.org/project/doped
.. image:: https://img.shields.io/conda/vn/conda-forge/doped.svg
   :target: https://anaconda.org/conda-forge/doped
.. image:: https://img.shields.io/pypi/dm/doped
   :target: https://pypi.org/project/doped

.. image:: docs/doped_v2_logo.png
   :align: right
   :width: 300

``doped`` is a python package for managing solid-state defect calculations, with functionality to
generate defect structures and relevant competing phases (for chemical potentials), interface with `ShakeNBreak`_ for defect structure-searching, write VASP input files for defect supercell calculations, and automatically parse and analyse the results.

Tutorials showing the code functionality and usage are provided on the `docs`_ site.

Example Outputs:
-----------------
Chemical potential/stability region plots and defect formation energy (a.k.a. transition level) diagrams:

.. image:: docs/doped_chempot_plotting.png
   :align: left
   :width: 420
.. image:: docs/doped_TLD_plot.png
   :align: right
   :width: 390

Installation
--------------

.. code-block:: bash

   pip install doped  # install doped and dependencies

Alternatively if desired, `doped` can also be installed from `conda` with:

.. code-block:: bash

   conda install -c conda-forge doped

Setup `POTCAR`s and `Materials Project` API
--------------------------------------------

To generate `VASP` `POTCAR` input files, and auto-determine `INCAR` settings such as `NELECT` for charged defects, your `POTCAR` directory needs to be setup to work with `pymatgen` (via the `.pmgrc.yaml` file).

**Instructions:**

#.

   - Set the VASP pseudopotential directory and your Materials Project API key in ``$HOME/.pmgrc.yaml`` (`pymatgen` config file) as follows:

     .. code-block:: bash

        PMG_VASP_PSP_DIR: <Path to VASP pseudopotential top directory>
        PMG_MAPI_KEY: <Your MP API key obtained from https://legacy.materialsproject.org/open>

    #. [The rest of your instructions go here.]

`ShakeNBreak`
---------------

As shown in the example notebook, it is highly recommended to use the `ShakeNBreak`_ approach when calculating point defects in solids. [Truncated for brevity.]

Summary GIF:
.. image:: docs/SnB_Supercell_Schematic_PES_2sec_Compressed.gif

`SnB` CLI Usage:
.. image:: docs/SnB_CLI.gif

Requirements
-------------

``doped`` requires ``pymatgen>=2022.10.22`` and its dependencies.

Developer Installation
------------------------

If you want to use the `example files`_ from the tutorials, [Instructions go here.]

Acknowledgments
------------------

`doped` (née `DefectsWithTheBoys` #gonebutnotforgotten) has benefitted from feedback [Truncated for brevity.]

The colour scheme for defect formation energy plots was originally templated from `AIDE`_.

.. _ShakeNBreak: https://shakenbreak.readthedocs.io
.. _docs: #
.. _example files: examples
.. _AIDE: https://github.com/SMTG-UCL/aide
