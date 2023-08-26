doped
=========

.. image:: https://github.com/SMTG-UCL/doped/actions/workflows/build_and_test.yml/badge.svg
   :target: https://github.com/SMTG-UCL/doped/actions
.. image:: https://readthedocs.org/projects/doped/badge/?version=latest&style=flat
   :target: https://doped.readthedocs.io/en/latest/
.. image:: https://img.shields.io/pypi/v/doped
   :target: https://pypi.org/project/doped
.. image:: https://img.shields.io/conda/vn/conda-forge/doped.svg
   :target: https://anaconda.org/conda-forge/doped
.. image:: https://img.shields.io/pypi/dm/doped
   :target: https://pypi.org/project/doped

.. image:: doped_v2_logo.png
   :align: right
   :width: 250


``doped`` is a Python package for managing solid-state defect calculations, with functionality to
generate defect structures and relevant competing phases (for chemical potentials), interface with
`ShakeNBreak`_ for defect structure-searching, write VASP input files for defect supercell calculations,
and automatically parse and analyse the results.

Example Outputs:
-----------------
Chemical potential/stability region plots and defect formation energy (a.k.a. transition level) diagrams:

.. image:: doped_chempot_plotting.png
   :align: left
   :width: 305
.. image:: doped_TLD_plot.png
   :align: right
   :width: 325

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

Tutorials showing the code functionality and usage are provided on the :ref:`Tutorials` page.

``doped`` can be installed via PyPI (``pip install doped``) or ``conda`` if preferred, and further
instructions for setting up ``POTCAR`` files with ``pymatgen`` (needed for input file generation), if not
already done, are provided on the :ref:`Installation` page.


``ShakeNBreak``
================
As shown in the tutorials, it is highly recommended to use the `ShakeNBreak`_ approach when calculating
point defects in solids, to ensure you have identified the ground-state structures of your defects. As
detailed in the `theory paper`_, skipping this step can result in drastically incorrect formation
energies, transition levels, carrier capture (basically any property associated with defects). This
approach is followed in the :ref:`tutorials <Tutorials>`, with a more in-depth explanation and tutorial
given on
the
`ShakeNBreak`_ docs.

.. _theory paper: https://www.nature.com/articles/s41524-023-00973-1
.. _tutorials: https://www.nature.com/articles/s41524-023-00973-1 # TODO!!!

Summary GIF:

.. image:: https://raw.githubusercontent.com/SMTG-UCL/ShakeNBreak/main/docs/SnB_Supercell_Schematic_PES_2sec_Compressed.gif

``SnB`` CLI Usage:

.. image:: https://raw.githubusercontent.com/SMTG-UCL/ShakeNBreak/main/docs/SnB_CLI.gif

.. raw:: html

   <br>
   <br>
   <br>
   <br>

Acknowledgements
================
``doped`` (née ``DefectsWithTheBoys`` #iykyk) has benefitted from feedback from many
users, in particular members of the `Scanlon <http://davidscanlon.com/>`_ and
`Walsh <https://wmd-group.github.io/>`_ research groups who are using it in
their work. Direct contributors are listed in the GitHub ``Contributors`` sidebar; including Seán
Kavanagh, Bonan Zhu, Katarina Brlec, Adair Nicolson, Sabrine Hachmioune and Savya Aggarwal.

Code to efficiently identify defect species from input supercell structures was contributed by
`Alex Ganose`_, and the colour scheme for defect formation energy plots was originally templated from
the ``aide`` package, developed by `Adam Jackson`_ and `Alex Ganose`_.

The `docs`_ website setup was templated from the ``ShakeNBreak`` docs set up by `Irea Mosquera-Lois`_ 🙌

``doped`` was originally based on the excellent `PyCDT`_ (no longer maintained), but transformed and
morphed over time as more and more functionality was added. After breaking changes in ``pymatgen``, the
package was rewritten to operate using the new ``pymatgen-analysis-defects`` package.

Studies using ``doped``
========================

- A\. T. J. Nicolson et al. `Journal of Materials Chemistry A <https://doi.org/10.1039/D3TA02429F>`__ 2023
- Y\. W. Woo, Z. Li, Y-K. Jung, J-S. Park, A. Walsh `ACS Energy Letters <https://doi.org/10.1021/acsenergylett.2c02306>`__ 2023
- P\. A. Hyde et al. `Inorganic Chemistry <https://doi.org/10.1021/acs.inorgchem.3c01510>`_ 2023
- J\. Willis, K. B. Spooner, D. O. Scanlon. `ChemRxiv <https://chemrxiv.org/engage/chemrxiv/article-details/64c29140ce23211b20a787bb>`__ 2023
- X\. Wang et al. `arXiv`_ 2023
- J\. Cen et al. `Journal of Materials Chemistry A`_ 2023
- J\. Willis & R. Claes et al. `ChemRxiv <https://doi.org/10.26434/chemrxiv-2023-lttnf>`__ 2023
- I\. Mosquera-Lois & S. R. Kavanagh, A. Walsh, D. O. Scanlon `npj Computational Materials`_ 2023
- Y\. T. Huang & S. R. Kavanagh et al. `Nature Communications`_ 2022
- S\. R. Kavanagh, D. O. Scanlon, A. Walsh, C. Freysoldt `Faraday Discussions`_ 2022
- S\. R. Kavanagh, D. O. Scanlon, A. Walsh `ACS Energy Letters <https://pubs.acs.org/doi/full/10.1021/acsenergylett.1c00380>`__ 2021
- C\. J. Krajewska et al. `Chemical Science`_ 2021

.. CSTX JPCC (for setting up phase diagram calcs)
.. Kumagai PRX Energy
.. Se
.. Oba book
.. BiOI
.. Kumagai collab paper
.. Lavan LiNiO2
.. Sykes Magnetic oxide polarons
.. Kat YTOS
.. Squires (and mention benchmark test against AIRSS? See Slack message)

.. _arXiv: https://arxiv.org/abs/2302.04901
.. _Journal of Materials Chemistry A: https://doi.org/10.1039/D3TA00532A
.. _npj Computational Materials: https://www.nature.com/articles/s41524-023-00973-1
.. _Nature Communications: https://www.nature.com/articles/s41467-022-32669-3
.. _Faraday Discussions: https://doi.org/10.1039/D2FD00043A
.. _Chemical Science: https://doi.org/10.1039/D1SC03775G

.. _ShakeNBreak: https://shakenbreak.readthedocs.io
.. _docs: https://doped.readthedocs.io
.. _Irea Mosquera-Lois: https://scholar.google.com/citations?user=oIMzt0cAAAAJ&hl=en
.. _PyCDT: https://www.sciencedirect.com/science/article/pii/S0010465518300079
.. _Adam Jackson: https://github.com/ajjackson
.. _Alex Ganose: https://github.com/utf

.. raw:: html

    <!-- Default Statcounter code for doped docs
    https://doped.readthedocs.io -->
    <script type="text/javascript">
    var sc_project=12911549;
    var sc_invisible=1;
    var sc_security="2e6f5c70";
    </script>
    <script type="text/javascript"
    src="https://www.statcounter.com/counter/counter.js"
    async></script>
    <noscript><div class="statcounter"><a title="Web Analytics"
    href="https://statcounter.com/" target="_blank"><img
    class="statcounter"
    src="https://c.statcounter.com/12911549/0/2e6f5c70/1/"
    alt="Web Analytics"
    referrerPolicy="no-referrer-when-downgrade"></a></div></noscript>
    <!-- End of Statcounter Code -->

.. toctree::
   :hidden:
   :caption: Usage
   :maxdepth: 4

   Installation
   Python API <doped>
   Tutorials
   Tips
   Troubleshooting

.. toctree::
   :hidden:
   :caption: Information
   :maxdepth: 1

   Contributing
   Code_Compatibility
   changelog_link
   doped on GitHub <https://github.com/SMTG-UCL/doped>
