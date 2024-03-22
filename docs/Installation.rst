.. _installation:

Installation
==============

.. code-block:: bash

   pip install doped  # install doped and dependencies

Alternatively if desired, ``doped`` can also be installed from ``conda`` with:

.. code-block:: bash

   conda install -c conda-forge doped
   pip install pydefect  # pydefect not available on conda, so needs to be installed with pip or otherwise, if using the eFNV correction

If you are installing ``doped`` via ``conda`` and you plan on using the eFNV (Kumagai-Oba) finite-size
correction, you will need to install the ``pydefect`` package with ``pip`` as shown or otherwise, as it is
not available on ``conda``.

It may be desirable to install ``doped`` in a virtual envionment (e.g. if you encounter package dependency
conflict warnings during installation etc). You can do this with ``conda`` with:

.. code-block:: bash

   conda create -n doped python=3.11
   conda activate doped
   pip install doped


Setup ``POTCAR``\s and Materials Project API
--------------------------------------------

To generate ``VASP`` ``POTCAR`` input files, and auto-determine ``INCAR`` settings such as ``NELECT``
for charged defects, your ``POTCAR`` directory needs to be setup to work with ``pymatgen`` (via the
``~/.pmgrc.yaml`` file).

**Instructions:**

1. Set the ``VASP`` pseudopotential directory and your Materials Project API key in ``$HOME/.pmgrc.yaml``
(``pymatgen`` config file) as follows:

   .. code-block:: bash

      PMG_VASP_PSP_DIR: <Path to VASP pseudopotential top directory>
      PMG_MAPI_KEY: <Your MP API key obtained from https://legacy.materialsproject.org/open>

   Within your ``VASP pseudopotential top directory``, you should have a folder named
   ``POT_GGA_PAW_PBE``/``potpaw_PBE.54``/``POT_GGA_PAW_PBE_54`` which contains ``POTCAR.X(.gz)`` files,
   generated using ``pmg config``.

2. If you have not previously setup your ``POTCAR`` directory in this way with ``pymatgen``, then follow these steps:

   .. code-block:: bash

      mkdir temp_potcars  # make a top folder to store the unzipped POTCARs
      mkdir temp_potcars/POT_GGA_PAW_PBE  # make a subfolder to store the unzipped POTCARs
      mv potpaw_PBE.54.tar.gz temp_potcars/POT_GGA_PAW_PBE  # copy in your zipped VASP POTCAR source
      cd temp_potcars/POT_GGA_PAW_PBE
      tar -xf potpaw_PBE.54.tar.gz  # unzip your VASP POTCAR source
      cd ../..  # return to the top folder
      pmg config -p temp_potcars psp_resources  # configure the psp_resources pymatgen POTCAR directory
      pmg config --add PMG_VASP_PSP_DIR "${PWD}/psp_resources"  # add the POTCAR directory to pymatgen's config file ($HOME/.pmgrc.yaml)
      rm -r temp_potcars  # remove the temporary POTCAR directory

3. If this has been successful, you should be able to run the shell commands:

   .. code-block:: bash

      pmg potcar -s Na_pv
      grep PBE POTCAR

   Which should then show ``PAW_PBE Na_pv {date}`` as the output (you can ignore any ``pymatgen`` warnings
   about recognising the ``POTCAR`` type).

4. If this does not work, you may need to add this to the ``.pmgrc.yaml`` file:

   .. code-block:: yaml

      PMG_DEFAULT_FUNCTIONAL: PBE  # whatever functional label your POTCARs have

   Note the Materials Project API key is required for determining the necessary competing phases to calculate in order to determine the chemical potential limits (required for defect formation energies). This should correspond to the legacy MP API, with your unique key available at: https://legacy.materialsproject.org/open.


Developer Installation
-----------------------
If you want to use the example files from the tutorials or run the package tests, you will need to clone
the ``doped`` GitHub repository:

#. Download the ``doped`` source code using the command:

.. code-block:: bash

   git clone https://github.com/SMTG-Bham/doped

#. Navigate to root directory:

.. code-block:: bash

   cd doped

#. Install the code, using the command:

.. code-block:: bash

   pip install -e .

Requirements
-------------

``doped`` requires ``pymatgen>=2022.10.22`` and its dependencies.