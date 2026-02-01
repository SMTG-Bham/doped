.. _tutorials:

Tutorials
===========================================================

.. _tutorials-literature:

Literature
----------

- **YouTube Overview (14 mins)**: `doped: Python package for charged defect supercell calculations <https://www.youtube.com/watch?v=-Z-R9sedeqY>`__
    - **B站 (Bilibili), 有中文字幕**: `建模缺陷超胞与计算的Python软件包：doped <https://www.bilibili.com/list/6073855/?sid=4603908>`__
- **Code Paper**: S\. R. Kavanagh et al. `doped: Python toolkit for robust and repeatable charged defect supercell calculations <https://doi.org/10.21105/joss.06433>`__. *Journal of Open Source Software* 9 (96), 6433, **2024**
- **General Defect Modelling Tutorial Video**: `Modelling Point Defects in Semiconductors with VASP <https://www.youtube.com/watch?v=FWz7nm9qoNg&ab_channel=Se%C3%A1nR.Kavanagh>`__
    - **B站 (Bilibili), 有中文字幕**: `使用VASP理解与计算半导体中的缺陷 <https://www.bilibili.com/list/6073855/?sid=4603908&oid=113988666990435&bvid=BV1V5KVeYEMn>`__
- **ShakeNBreak Documentation**: |ShakeNBreakDocs|
- **Guidelines for Defect Simulations**: A\. G. Squires et al. `Guidelines for robust and reproducible point defect simulations in crystals <https://doi.org/10.26434/chemrxiv-2025-3lb5k>`__ **2025**

See the :ref:`Literature` section on the main page for further recommended literature on defect 
simulations.

.. note that oddly, the Bilibili links don't seem to work on MacOS Chrome (but do if you click the address
.. in the address bar and press enter, weird), and works fine on everything else...

Code Tutorials
--------------

The typical defect calculation, parsing and analysis workflow using ``doped`` is exemplified through the
tutorials:

.. toctree::
   :maxdepth: 2

   generation_tutorial
   parsing_tutorial
   thermodynamics_tutorial
   chemical_potentials_tutorial
   advanced_analysis_tutorial
   fermisolver_tutorial
   plotting_customisation_tutorial
   GGA_workflow_tutorial
   CCD_NEB_tutorial
