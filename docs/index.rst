``doped``
=========

.. image:: https://github.com/SMTG-Bham/doped/actions/workflows/test.yml/badge.svg
   :target: https://github.com/SMTG-Bham/doped/actions
.. image:: https://readthedocs.org/projects/doped/badge/?version=latest&style=flat
   :target: https://doped.readthedocs.io/en/latest/
.. image:: https://img.shields.io/pypi/v/doped
   :target: https://pypi.org/project/doped
.. image:: https://img.shields.io/conda/vn/conda-forge/doped.svg
   :target: https://anaconda.org/conda-forge/doped
.. image:: https://img.shields.io/pypi/dm/doped
   :target: https://pypi.org/project/doped
.. image:: https://joss.theoj.org/papers/10.21105/joss.06433/status.svg
   :target: https://doi.org/10.21105/joss.06433

.. raw:: html

   <img src="https://raw.githubusercontent.com/SMTG-Bham/doped/main/docs/doped_v2_logo.png" align="right" width="200" alt="Schematic of a doped (defect-containing) crystal, inspired by the biological analogy to (semiconductor) doping." title="Schematic of a doped (defect-containing) crystal, inspired by the biological analogy to (semiconductor) doping.">

``doped`` is a Python software for the generation, pre-/post-processing and analysis of defect supercell
calculations, implementing the defect simulation workflow in an efficient, reproducible, user-friendly yet
powerful and fully-customisable manner.

Tutorials showing the code functionality and usage are provided on the :ref:`Tutorials` page, and an
overview of the key advances of the package is given in the
`JOSS paper <https://doi.org/10.21105/joss.06433>`__.

.. raw:: html

    <a href="https://doi.org/10.21105/joss.06433"><img class="center" width="800" src="https://raw.githubusercontent.com/SMTG-Bham/doped/main/docs/JOSS/doped_JOSS_workflow_figure.png"></a>

Key Features
============
All features and functionality are fully-customisable:

- **Supercell Generation**: Generate an optimal supercell, maximising periodic image separation for the minimum number of atoms (computational cost).
- **Defect Generation**: Generate defect supercells and likely charge states from chemical intuition.
- **Calculation I/O**: Automatically write inputs & parse calculations (``VASP`` & other DFT/force-field codes).
- **Chemical Potentials**: Determine relevant competing phases for chemical potential limits, with automated calculation setup, parsing and analysis.
- **Defect Analysis**: Automatically parse calculation outputs to compute defect formation energies, finite-size corrections (FNV & eFNV), symmetries, degeneracies, transition levels, etc.
- **Thermodynamic Analysis**: Compute (non-)equilibrium Fermi levels, defect/carrier concentrations etc. as functions of annealing/cooling temperature, chemical potentials, full inclusion of metastable states etc.
- **Plotting**: Generate publication-quality plots of defect formation energies, chemical potential limits, defect/carrier concentrations, Fermi levels, charge corrections, etc.
- ``Python`` **Interface**: Fully-customisable and modular ``Python`` API, being plug-and-play with `ShakeNBreak`_ for `defect structure-searching <https://www.nature.com/articles/s41524-023-00973-1>`_, `easyunfold <https://smtg-bham.github.io/easyunfold/>`__ for band unfolding, `CarrierCapture.jl <https://github.com/WMD-group/CarrierCapture.jl>`__/`nonrad <https://nonrad.readthedocs.io/en/latest/>`__ for non-radiative recombination etc.
- Reproducibility, tabulation, automated compatibility/sanity checking, strain/displacement analysis, shallow defect / eigenvalue analysis, high-throughput compatibility, Wyckoff analysis...

Performance and Example Outputs
-------------------------------

.. image:: JOSS/doped_JOSS_figure.png
   :target: https://doi.org/10.21105/joss.06433

**(a)** Optimal supercell generation comparison. **(b)** Charge state estimation comparison.
Example **(c)** Kumagai-Oba (eFNV) finite-size correction plot, **(d)** defect formation energy diagram,
**(e)** chemical potential / stability region, **(f)** Fermi level vs. annealing temperature, **(g)**
defect/carrier concentrations vs. annealing temperature and **(h)** Fermi level / carrier concentration
heatmap plots from ``doped``. Automated plots of **(i,j)** single-particle eigenvalues and **(k)** site
displacements from DFT supercell calculations. See the
`JOSS paper <https://doi.org/10.21105/joss.06433>`__ for more details.

Installation
------------
``doped`` can be installed via PyPI (``pip install doped``) or ``conda`` if preferred, and further
instructions for setting up ``POTCAR`` files with ``pymatgen`` (needed for input file generation), if not
already done, are provided on the :ref:`Installation` page.

Citation
========

If you use ``doped`` in your research, please cite:

- S\. R. Kavanagh et al. `doped: Python toolkit for robust and repeatable charged defect supercell calculations <https://doi.org/10.21105/joss.06433>`__. *Journal of Open Source Software* 9 (96), 6433, **2024**

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

.. image:: https://raw.githubusercontent.com/SMTG-Bham/ShakeNBreak/main/docs/SnB_Supercell_Schematic_PES_2sec_Compressed.gif

Studies using ``doped``, so far
===============================

- X\. Wang et al. **Upper efficiency limit of Sb₂Se₃ solar cells** `arXiv <https://arxiv.org/abs/2402.04434>`_ 2024
- I\. Mosquera-Lois et al. **Machine-learning structural reconstructions for accelerated point defect calculations** `arXiv <https://doi.org/10.48550/arXiv.2401.12127>`_ 2024
- W\. Dou et al. **Giant Band Degeneracy via Orbital Engineering Enhances Thermoelectric Performance from Sb₂Si₂Te₆ to Sc₂Si₂Te₆** `ChemRxiv <https://doi.org/10.26434/chemrxiv-2024-hm6vh>`_ 2024
- K\. Li et al. **Computational Prediction of an Antimony-based n-type Transparent Conducting Oxide: F-doped Sb₂O₅** `Chemistry of Materials <https://doi.org/10.1021/acs.chemmater.3c03257>`_ 2024
- X\. Wang et al. **Four-electron negative-U vacancy defects in antimony selenide** `Physical Review B <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.108.134102>`_ 2023
- Y\. Kumagai et al. **Alkali Mono-Pnictides: A New Class of Photovoltaic Materials by Element Mutation** `PRX Energy <http://dx.doi.org/10.1103/PRXEnergy.2.043002>`__ 2023
- S\. M. Liga & S. R. Kavanagh, A. Walsh, D. O. Scanlon, G. Konstantatos **Mixed-Cation Vacancy-Ordered Perovskites (Cs₂Ti** :sub:`1–x` **Sn** :sub:`x` **X₆; X = I or Br): Low-Temperature Miscibility, Additivity, and Tunable Stability** `Journal of Physical Chemistry C`_ 2023
- A\. T. J. Nicolson et al. **Cu₂SiSe₃ as a promising solar absorber: harnessing cation dissimilarity to avoid killer antisites** `Journal of Materials Chemistry A <https://doi.org/10.1039/D3TA02429F>`__ 2023
- Y\. W. Woo, Z. Li, Y-K. Jung, J-S. Park, A. Walsh **Inhomogeneous Defect Distribution in Mixed-Polytype Metal Halide Perovskites** `ACS Energy Letters <https://doi.org/10.1021/acsenergylett.2c02306>`__ 2023
- P\. A. Hyde et al. **Lithium Intercalation into the Excitonic Insulator Candidate Ta₂NiSe₅** `Inorganic Chemistry <https://doi.org/10.1021/acs.inorgchem.3c01510>`_ 2023
- J\. Willis, K. B. Spooner, D. O. Scanlon. **On the possibility of p-type doping in barium stannate** `Applied Physics Letters <https://doi.org/10.1063/5.0170552>`__ 2023
- J\. Cen et al. **Cation disorder dominates the defect chemistry of high-voltage LiMn** :sub:`1.5` **Ni** :sub:`0.5` **O₄ (LMNO) spinel cathodes** `Journal of Materials Chemistry A`_ 2023
- J\. Willis & R. Claes et al. **Limits to Hole Mobility and Doping in Copper Iodide** `Chem Mater <https://doi.org/10.1021/acs.chemmater.3c01628>`__ 2023
- I\. Mosquera-Lois & S. R. Kavanagh, A. Walsh, D. O. Scanlon **Identifying the ground state structures of point defects in solids** `npj Computational Materials`_ 2023
- Y\. T. Huang & S. R. Kavanagh et al. **Strong absorption and ultrafast localisation in NaBiS₂ nanocrystals with slow charge-carrier recombination** `Nature Communications`_ 2022
- S\. R. Kavanagh, D. O. Scanlon, A. Walsh, C. Freysoldt **Impact of metastable defect structures on carrier recombination in solar cells** `Faraday Discussions`_ 2022
- Y-S\. Choi et al. **Intrinsic Defects and Their Role in the Phase Transition of Na-Ion Anode Na₂Ti₃O₇** `ACS Appl. Energy Mater. <https://doi.org/10.1021/acsaem.2c03466>`__ 2022
- S\. R. Kavanagh, D. O. Scanlon, A. Walsh **Rapid Recombination by Cadmium Vacancies in CdTe** `ACS Energy Letters <https://pubs.acs.org/doi/full/10.1021/acsenergylett.1c00380>`__ 2021
- C\. J. Krajewska et al. **Enhanced visible light absorption in layered Cs₃Bi₂Br₉ through mixed-valence Sn(II)/Sn(IV) doping** `Chemical Science`_ 2021

.. Se
.. Oba book
.. BiOI
.. Kumagai collab paper
.. Lavan LiNiO2
.. Sykes Magnetic oxide polarons
.. Kat YTOS
.. Squires (and mention benchmark test against AIRSS? See Slack message)

.. _Journal of Physical Chemistry C: https://doi.org/10.1021/acs.jpcc.3c05204
.. _Journal of Materials Chemistry A: https://doi.org/10.1039/D3TA00532A
.. _npj Computational Materials: https://www.nature.com/articles/s41524-023-00973-1
.. _Nature Communications: https://www.nature.com/articles/s41467-022-32669-3
.. _Faraday Discussions: https://doi.org/10.1039/D2FD00043A
.. _Chemical Science: https://doi.org/10.1039/D1SC03775G

Acknowledgements
================

``doped`` (née `DefectsWithTheBoys` #iykyk) has benefitted from feedback from many users, in particular
members of the `Scanlon <http://davidscanlon.com/>`_ and
`Walsh <https://wmd-group.github.io/>`_ research groups who have / are using it in their work.
Direct contributors are listed in the GitHub ``Contributors`` sidebar; including Seán Kavanagh,
Alex Squires, Adair Nicolson, Irea Mosquera-Lois, Alex Ganose, Bonan Zhu, Katarina Brlec, Sabrine
Hachmioune and Savya Aggarwal.

`doped` was originally based on the excellent ``PyCDT`` (no longer maintained), but transformed and morphed
over time as more and more functionality was added. After breaking changes in ``pymatgen``, the package
was entirely refactored and rewritten, to work with the new ``pymatgen-analysis-defects`` package.

.. _ShakeNBreak: https://shakenbreak.readthedocs.io

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
   doped on GitHub <https://github.com/SMTG-Bham/doped>
