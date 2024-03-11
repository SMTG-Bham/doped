---
title: 'doped: Python toolkit for robust and repeatable charged defect supercell calculations'
tags:
  - Python
  - materials modelling
  - materials physics
  - materials chemistry
  - thermodynamics
  - point defects
  - DFT
  - VASP
  - pymatgen
  - semiconductors
  - ab initio
  - structure searching
  - materials science
  - finite-size corrections
  - supercell calculations
authors:
  - name: Seán R. Kavanagh
    corresponding: true
    affiliation: "1, 2"
    orcid: 0000-0003-4577-9647
  - name: Alexander G. Squires 
    orcid: 0000-0001-6967-3690
    affiliation: 3
  - name: Adair Nicolson
    orcid: 0000-0002-8889-9369
    affiliation: 2
  - name: Irea Mosquera-Lois
    orcid: 0000-0001-7651-0814
    affiliation: 1
  - name: Alex M. Ganose
    orcid: 0000-0002-4486-3321
    affiliation: 4
  - name: Bonan Zhu
    orcid: 0000-0001-5601-6130
    affiliation: 2
  - name: Katarina Brlec
    orcid: 0000-0003-1485-1888
    affiliation: 2
  - name: Aron Walsh
    affiliation: 1
    orcid: 0000-0001-5460-7033
  - name: David O. Scanlon
    affiliation: 3
    orcid: 0000-0001-9174-8601
affiliations:
 - name: Thomas Young Centre and Department of Materials, Imperial College London, United Kingdom
   index: 1
 - name: Thomas Young Centre and Department of Chemistry, University College London, United Kingdom
   index: 2
 - name: School of Chemistry, University of Birmingham, Birmingham, United Kingdom
   index: 3
 - name: Department of Chemistry, Imperial College London, London, United Kingdom
   index: 4
date: 01 March 2024
bibliography: paper.bib
---

# Summary

Defects are a universal feature of crystalline solids, dictating the key properties and performance of many functional materials. 
Given their crucial importance yet inherent difficulty in measuring experimentally, computational methods (such as DFT and ML/classical force-fields) are widely used to predict defect behaviour at the atomic level and the resultant impact on macroscopic properties.
Here we report ``doped``, a Python package for the generation, pre-/post-processing and analysis of defect supercell calculations. 
``doped`` has been built to implement the defect simulation workflow in an efficient, user-friendly yet powerful and fully-flexible manner, with the goal of providing a robust general-purpose platform for conducting reproducible calculations of solid-state defect properties.

[//]: # (such as conductivity, carrier recombination, catalytic activity etc)
[//]: # (The typically dilute concentration of defects, despite their major impact on macroscopic properties, renders their experimental characterisation extremely challenging however. )
[//]: # (Thus, theoretical methods represent the primary avenue for the investigation of defects at the atomic-scale, with computational predictions of defect behaviour &#40;and resultant impact&#41; often being compared to experimental measurements of global properties, such as carrier concentrations, solar cell efficiency, ionic conductivity or catalytic activity.)
[//]: # (What's the sell?)
[//]: # (- Easy-to-use functionality to expedite the analysis workflow and facilitate powerful/in-depth defect analysis.)
[//]: # (Also compatible for high-throughput etc)

# Statement of need

The materials science sub-field of computational defect modelling has seen considerable growth in recent years, driven by the crucial importance of these species in functional materials and the major advances in computational methodologies and resources facilitating their accurate simulation.
Software which enables researchers to efficiently and accurately perform these calculations, while allowing for in-depth target analyses of the resultant data, is thus of significant value to the community.
Indeed there are many critical stages in the computational workflow for defects, which when performed manually not only consume significant researcher time and effort but also leave room for human error – particularly for newcomers to the field.
Moreover, there are growing efforts to perform high-throughput investigations of defects in solids [@xiong_high-throughput_2023], necessitating robust, user-friendly and efficient software implementing this calculation workflow.

[//]: # (By expediting the defect simulation workflow and providing efficient analysis tools, ``doped`` aims to... facilitate the investigation of defects in solids, and to enable the efficient and reproducible calculation of solid-state defect properties.)

Given this importance of defect simulations and the complexity of the workflow, a number of software packages have been developed with the goal of managing pre- and post-processing of defect calculations, including work on the `HADES`/`METADISE` codes from the 1970s [@parker_hades_2004], to more recent work from @Kumagai2021, @Broberg2018, @Shen2024, @neilson_defap_2022, @Arrigoni2021, @Goyal2017, @Huang2022, @pean_presentation_2017 and @naik_coffee_2018.[^1]
While each of these codes have their strengths, they do not include the full suite of functionality provided by `doped` – some of which is discussed below – nor adopt the same focus on user-friendliness (along with sanity-checking warnings & error catching) and efficiency with full flexibility and wide-ranging functionality, targeting expert-level users and newcomers to the field alike.

[^1]: Some of these packages are no longer maintained, not compatible with high-throughput architectures, and/or are closed-source/commercial packages.

[//]: # (Do we even need to say why `doped` is better?)

[//]: # (- some not maintained, not compatible with latest `pymatgen`, not compatible with high-throughput frameworks)

[//]: # (- DASP isn't open access and they're trying to commercialise)

![Schematic workflow of a computational defect investigation using `doped`. \label{fig_workflow}](doped_JOSS_workflow_figure.png)

# doped

`doped` is a Python software for the generation, pre-/post-processing and analysis of defect supercell calculations, as depicted in \autoref{fig_workflow}.
The design philosophy of `doped` has been to implement the defect simulation workflow in an efficient, reproducible, user-friendly yet powerful and fully-customisable manner, combining reasonable defaults with full user control for each parameter in the workflow.
As depicted in \autoref{fig_workflow}, the core functionality of `doped` is the generation of defect supercells and competing phases, writing calculation input files, parsing calculation outputs and analysing/plotting defect-related properties. This functionality and recommended usage of `doped` is demonstrated in the [tutorials](https://doped.readthedocs.io/en/latest/Tutorials.html) on the [documentation website](https://doped.readthedocs.io/en/latest/).

![**a.** Average minimum periodic image distance, normalised by the ideal image distance at that volume (i.e. for a perfect close-packed face-centred cubic (FCC) cell), versus the number of primitive unit cells, for the supercell generation algorithms of `doped`, `ASE` and `pymatgen`. "SC" = simple cubic and "HCP" = hexagonal close-packed. **b.** Average performance of charge state guessing routines from `doped` compared to alternative approaches, in terms of false positives and false negatives. Asterisk indicates that the `pyCDT` false _negatives_ are underestimated as the majority of this test set used the guessed charge state ranges from `pyCDT`. "Ox." = oxidation & "prob." = probabilities. Example **(c)** Kumagai-Oba (extended Freysoldt-Neugebauer-Van-de-Walle; "eFNV") finite-size correction plot, **(d)** defect formation energy diagram, **(e)** chemical potential / stability region, **(f)** Fermi level vs. annealing temperature, **(g)** defect/carrier concentrations vs. annealing temperature and **(h)** Fermi level / carrier concentration heatmap plots from `doped`. Data and code to reproduce these plots is provided in the [`docs/JOSS`](https://github.com/SMTG-Bham/doped/blob/main/docs/JOSS) subfolder of the `doped` GitHub repository. \label{fig1}](doped_JOSS_figure.png)

Some key advances of `doped` include:

- **Supercell Generation:** When choosing a simulation supercell for charged defects in materials, we typically want to maximise the minimum distance between periodic images of the defect (to reduce finite-size errors) while keeping the supercell to a tractable number of atoms/electrons to calculate. Common approaches are to choose a near-cubic integer expansion of the unit cell [@ong_python_2013], or to use a cell shape metric to search for optimal supercells [@larsen_atomic_2017]. Building on these and instead integrating an efficient algorithm for calculating minimum image distances, `doped` directly optimises the supercell choice for this goal – often identifying non-trivial 'root 2'/'root 3' type supercells. As illustrated in \autoref{fig1}a, this leads to a significant reduction in the supercell size (and thus computational cost) required to achieve a threshold minimum image distance.
  - Over a test set of simple cubic, trigonal, orthorhombic, monoclinic and face-centred cubic unit cells, the `doped` algorithm is found to give mean improvements of 35.2%, 9.1% and 6.7% in the minimum image distance for a given (maximum) number of unit cells as compared to the `pymatgen` cubic supercell algorithm, the `ASE` optimal cell shape algorithm with simple-cubic target shape, and `ASE` with FCC target shape respectively – in the range of 2-20 unit cells. For 2-50 unit cells (for which the mean values across this test set are plotted in \autoref{fig1}a), this becomes 36.0%, 9.3% and 5.6% respectively. Given the approximately cubic scaling of DFT computational cost with the number of atoms, these correspond to significant reductions in cost (~20-150%).
  - As always, the user has full control over supercell generation in `doped`, with the ability to specify/adjust constraints on the minimum image distance, number of atoms or transformation matrix, or to simply provide a pre-generated supercell if desired.
- **Charge-state guessing:** Defects in solids can adopt various electronic charge states. However, the set of stable charge states for a given defect is typically not known _a priori_ and so one must choose a set of _possible_ defect charge states to calculate – usually relying on some form of chemical intuition. In this regard, extremal defect charge states that are calculated but do not end up being stable can be considered 'false positives' or 'wasted' calculations,[^2] while charge states which are stable but were not calculated can be considered 'false negatives' or 'missed' calculations. `doped` builds on other routines which use known elemental oxidation states to additionally account for oxidation state _probabilities_, the electronic state of the host crystal and charge state magnitudes. Implementing these features in a simple cost function, we find a significant improvement in terms of both efficiency (reduced false positives) and completeness (reduced false negatives) for this charge state guessing step, as shown in \autoref{fig1}b.[^3]
  - Again, this step is fully-customisable. The user can tune the probability threshold at which to include charge states or manually specify defect charge states. All probability factors computed are available to the user and saved to the defect `JSON` files for full reproducibility.
  
[^2]: Note that _unstable_ defect charge states which are intermediate between _stable_ charge states (e.g. $X^0$ for a defect $X$ with a (+1/-1) negative-U level) should still be calculated and are _not_ considered false positives.

[//]: # (Because they're intermediate states for carrier capture etc – don't need to explain this)

[^3]: Given sufficient data, a machine learning model could likely further improve the performance of this charge state guessing step.

- **Efficient competing phase selection:** Elemental chemical potentials (a key term in the defect formation energy) are limited by the secondary phases which border the host compound on the phase diagram. These bordering phases are known as competing phases, and their total energies must be calculated to determine the chemical potential limits. Only the elemental reference phases and compounds which border the host on the phase diagram need to be calculated, rather than the full phase diagram.

  `doped` aims to improve the efficiency of this step by querying the [Materials Project](https://materialsproject.org) database (containing both experimentally-measured and theoretically-predicted crystal structures), and pulling only compounds which _could border the host material_ within a user-specified error tolerance for the semi-local DFT database energies (0.1 eV/atom by default), along with the elemental reference phases. The necessary _k_-point convergence step for these compounds is also implemented in a semi-automated fashion to expedite this process.
  - With the parsed chemical potentials in `doped`, the user can easily select various X-poor/rich chemical conditions, or scan over a range of chemical potentials (growth conditions) as shown in \autoref{fig1}e,h.

- **Automated symmetry & degeneracy handling:** `doped` automatically determines the point symmetry of both initial (un-relaxed) and final (relaxed) defect configurations, and computes the corresponding orientational (and spin) degeneracy factors. This is a key pre-factor in the defect concentration equation:

  \begin{equation}
  N_D = gN_s \exp(-E_f/k_BT)
  \end{equation}

  where $g$ is the product of all degeneracy factors, $N_s$ is the concentration of lattice sites for that defect, $E_f$ is the defect formation energy and $N_D$ is the defect concentration. $g$ can affect predicted defect/carrier concentrations by up to 2/3 orders of magnitude [@mosquera-lois_imperfections_2023; @kavanagh_impact_2022], and is often overlooked in defect calculations, partly due to the (previous) requirement of significant manual effort and knowledge of group theory.

- **Automated compatibility checking:** When parsing defect calculations, `doped` automatically checks that calculation parameters which could affect the defect formation energy (e.g. _k_-point grid, energy cutoff, pseudopotential choice, exchange fraction, Hubbard U etc.) are consistent between the defect and reference calculations. This is a common source of accidental error in defect calculations, and `doped` provides informative warnings if any inconsistencies are detected. 

- **Thermodynamic analysis:** `doped` provides a suite of flexible tools for the analysis of defect thermodynamics, including formation energy diagrams (\autoref{fig1}d), equilibrium & non-equilibrium Fermi level solving (\autoref{fig1}f), doping analysis (\autoref{fig1}g,h), Brouwer-type diagrams etc. These include physically-motivated (but tunable) grouping of defect sites, full inclusion of metastable states, support for complex system constraints, optimisation over high-dimensional chemical & temperature space and highly-customisable plotting.

- **Finite-size corrections:** Both the isotropic Freysoldt (FNV) [@Freysoldt2009] and anisotropic Kumagai (eFNV) [@kumagai_electrostatics-based_2014] image charge corrections are implemented automatically in `doped`, with tunable sampling radii / sites (which may be desirable for e.g. layered materials), automated correction plotting (to visualise/analyse convergence; \autoref{fig1}c) and automatic sampling error estimation.

- **Reproducibility & tabulation:** `doped` has been built to support and encourage reproducibility, with all input parameters and calculation results saved to lightweight `JSON` files. This allows for easy sharing of calculation inputs/outputs and reproducible analysis. Several tabulation functions are also provided to facilitate the quick summarising of key quantities as exemplified in the [tutorials](https://doped.readthedocs.io/en/latest/Tutorials.html) (including defect formation energy contributions, charge transition levels (with/without metastable states), symmetry, degeneracy and multiplicity factors, defect/carrier concentrations, chemical potential limits, dopability limits, doping windows...) to aid transparency, reproducibility, comparisons with other works and general analysis. The use of these tabulated outputs in supporting information of publications is encouraged.

- **High-throughput compatibility:** `doped` is built to be compatible with high-throughput architectures such as [atomate(2)](https://github.com/materialsproject/atomate2) [@atomate] or [AiiDA](https://aiida.net) [@AiiDA], aided by its object-oriented Python framework, JSON-serializable classes and sub-classed `pymatgen` objects. Examples are provided on the [documentation website](https://doped.readthedocs.io/en/latest/).

- **[`ShakeNBreak`](https://shakenbreak.readthedocs.io):** `doped` is natively interfaced with our defect structure-searching code `ShakeNBreak` [@mosquera-lois_shakenbreak_2022], seamlessly incorporating this phase in the defect calculation workflow. This step can optionally be skipped or an alternative structure-searching approach readily implemented.

Some additional features of `doped` include directional-dependent site displacement (local strain) analysis, deterministic & informative defect naming, molecule generation for gaseous competing phases, multiprocessing for expedited generation & parsing, shallow defect analysis (via `pydefect` [@Kumagai2021]), Wyckoff site analysis (including arbitrary/interstitial sites), controllable defect site placement to aid visualisation and more.

The defect generation and thermodynamic analysis components of `doped` are agnostic to the underlying software used for the defect supercell calculations.
Direct calculation I/O is fully-supported for `VASP` [@vasp], while input defect structure files can be generated for several widely-used DFT codes, including `FHI-aims` [@fhi_aims], `CP2K` [@cp2k], `Quantum Espresso` [@espresso] and `CASTEP` [@castep] via the `pymatgen` `Structure` object. Full support for calculation I/O with other DFT codes may be added in the future if there is sufficient demand.
Moreover, `doped` is built to be readily compatible with other computational toolkits for advanced defect characterisation, such as `ShakeNBreak` for defect structure-searching, `py-sc-fermi` for advanced thermodynamic analysis under complex constraints [@squires_py-sc-fermi_2023], `easyunfold` for analysing defect/dopant-induced electronic structure changes [@zhu_easyunfold_2024] or `CarrierCapture.jl`/`nonrad` for non-radiative recombination calculations [@kim_carriercapturejl_2020; @turiansky_nonrad_2021]. 

`doped` has been used to manage the defect simulation workflow in a number of publications thus far, including @wang_upper_2024, @cen_cation_2023, @nicolson_cu2sise3_2023, @li_computational_2023, @kumagai_alkali_2023, @woo_inhomogeneous_2023,  @wang_four-electron_2023-1, @mosquera-lois_search_2021, @mosquera-lois_identifying_2023, @mosquera-lois_machine-learning_2024, @huang_strong_2022, @dou_giant_2024, @liga_mixed-cation_2023, @willis_possibility_2023, @willis_limits_2023, @krajewska_enhanced_2021, @kavanagh_rapid_2021, @kavanagh_frenkel_2022.

# CRediT Author Contributions
**Seán R. Kavanagh:** Conceptualisation, Methodology, Software, Writing, Project Administration. **Alex G. Squires:** Code for complex doping analysis. **Adair Nicolson:** Code for shallow defect analysis. **Irea Mosquera-Lois:** Code for local strain analysis. **Katarina Brlec:** Competing phases code refactoring. **Aron Walsh & David Scanlon:** Funding Acquisition, Management, Ideas & Discussion. **All authors:** Feedback, Code Contributions, Writing – Review & Editing.

# Acknowledgements

`doped` has benefited from feature requests and feedback from many members of the Walsh and Scanlon research groups, including (but not limited to) Xinwei Wang, Sabrine Hachmioune, Savya Aggarwal, Daniel Sykes, Chris Savory, Jiayi Cen, Lavan Ganeshkumar, Ke Li, Kieran Spooner and Luisa Herring-Rodriguez.
S.R.K thanks Dr. Christoph Freysoldt and Prof. Yu Kumagai for useful discussions regarding the implementation of image charge corrections.

The initial development of `doped` was inspired by the `pyCDT` package from @Broberg2018, while the original colour scheme for defect formation energy plots was inspired by work from Drs. Adam J. Jackson and Alex M. Ganose.
`doped` makes extensive use of Python objects from the widely-used `pymatgen` [@ong_python_2013] package (such as structure representations and VASP I/O handling), as well as crystal symmetry functions from `spglib` [@togo_textttspglib_2018].

S.R.K. and A.N. acknowledge the EPSRC Centre for Doctoral Training in the Advanced
Characterisation of Materials (CDTACM)(EP/S023259/1) for funding PhD studentships. DOS acknowledges
support from the EPSRC (EP/N01572X/1) and from the European Research Council, ERC (Grant No. 758345).
The PRAETORIAN project was funded by UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee (EP/Y019504/1). This work used the ARCHER2 UK National Supercomputing Service (https://www.archer2.ac.uk), via our membership of the UK’s HEC Materials Chemistry Consortium, which is funded by the EPSRC (EP/L000202, EP/R029431 and EP/T022213), the UK Materials and Molecular Modelling (MMM) Hub (Young EP/T022213).

# References