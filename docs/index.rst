doped
=========

.. image:: https://github.com/SMTG-UCL/doped/actions/workflows/build_and_test.yml/badge.svg
   :target: https://github.com/SMTG-UCL/doped/actions
.. image:: https://img.shields.io/pypi/v/doped
   :target: https://pypi.org/project/doped
.. image:: https://img.shields.io/conda/vn/conda-forge/doped.svg
   :target: https://anaconda.org/conda-forge/doped
.. image:: https://img.shields.io/pypi/dm/doped
   :target: https://pypi.org/project/doped

.. image:: doped_v2_logo.png
   :align: right
   :width: 275


``doped`` is a python package for managing solid-state defect calculations, with functionality to
generate defect structures and relevant competing phases (for chemical potentials), interface with
`ShakeNBreak`_ for defect structure-searching, write VASP input files for defect supercell calculations,
and automatically parse and analyse the results.

Example Outputs:
-----------------
Chemical potential/stability region plots and defect formation energy (a.k.a. transition level) diagrams:

.. image:: doped_chempot_plotting.png
   :align: left
   :width: 365
.. image:: doped_TLD_plot.png
   :align: right
   :width: 385

.. raw:: html

   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>
   <br>

Tutorials showing the code functionality and usage are provided on the `docs`_ site.
``doped`` can be installed via ``pip`` or ``conda``, and further instructions for setting up ``POTCAR``s
with ``pymatgen`` (needed for input file generation), if not already done, are provided on
the `Installation`_ page.


``ShakeNBreak``
================
As shown in the tutorials, it is highly recommended to use the `ShakeNBreak`_ approach when calculating
point defects in solids, to ensure you have identified the groundstate structures of your defects. As
detailed in the `theory paper`_, skipping this step can result in drastically incorrect formation
energies, transition levels, carrier capture (basically any property associated with defects). This
approach is followed in the `tutorials`_, with a more in-depth explanation and tutorial given on the
`ShakeNBreak`_ docs.

.. _theory paper: https://www.nature.com/articles/s41524-023-00973-1
.. _tutorials: https://www.nature.com/articles/s41524-023-00973-1 # TODO!!!

Summary GIF:

.. image:: SnB_Supercell_Schematic_PES_2sec_Compressed.gif

``SnB`` CLI Usage:

.. image:: SnB_CLI.gif

.. raw:: html

   <br>
   <br>
   <br>
   <br>

Acknowledgments
================
``doped`` (née ``DefectsWithTheBoys`` #gonebutnotforgotten) has benefitted from feedback from many
users, in particular members of the Walsh and Scanlon research groups who have used / are using it in
their work. Direct contributors are listed in the GitHub ``Contributors`` sidebar; including Seán
Kavanagh, Bonan Zhu, Katarina Brlec, Adair Nicolson, Sabrine Hachmioune and Savya Aggarwal.

Code to efficiently identify defect species from input supercell structures was contributed by Dr `Alex Ganose`_.

``doped`` was originally based on the excellent `PyCDT`_ (no longer maintained), but transformed and
morphed over time as more and more functionality was added. After breaking changes in ``pymatgen``, the
package was rewritten to operate using the new ``pymatgen-analysis-defects`` package.

The colour scheme for defect formation energy plots was originally templated from ``aide`` (#neverforget),
developed by the dynamic duo `Adam Jackson`_ and `Alex Ganose`_.


Studies using ``doped``
========================

**TODO**
We'll add papers that use `doped` to this list as they come out!

.. _ShakeNBreak: https://shakenbreak.readthedocs.io
.. _docs: #
.. _example files: examples
.. _PyCDT: https://www.sciencedirect.com/science/article/pii/S0010465518300079
.. _AIDE: https://github.com/SMTG-UCL/aide
.. _Adam Jackson: https://github.com/ajjackson
.. _Alex Ganose: https://github.com/utf

.. toctree::
   :hidden:
   :maxdepth: 4

   Installation
   Python API <doped>
   Tutorials
   Contributing
