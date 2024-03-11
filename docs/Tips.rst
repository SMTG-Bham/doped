Tips & Tricks
============================

The development philosophy behind ``doped`` has been to build a powerful, efficient and flexible code
for managing and analysing solid-state defect calculations, having reasonable defaults (that work well for
the majority of materials/defects) with flexibility for the user to customise the workflow to
their specific needs/system.

.. note::
    While we provide some general rules-of-thumb for reasonable choices in the calculation workflow
    (based on the literature and our experience), there is no substitute for the user's own judgement.
    Defect behaviour is system-dependent, so it is `always` important to question and
    consider the choices and approximations made in the workflow (such as supercell choice, charge state
    ranges, interstitial site pruning, `MAGMOM` initialisation etc.) in the context of your specific
    host system.

Interstitials
-------------------
As described in the `YouTube defect calculation tutorial <https://youtu.be/FWz7nm9qoNg>`_, our
recommended workflow for calculating interstitial defects is to first generate the set of
candidate interstitial sites for your structure using ``DefectsGenerator`` (which uses Voronoi tessellation
for this, see note below), and then perform Gamma-point-only relaxations (using ``vasp_gam``) for each
charge state of the generated interstitial candidates, and then pruning some of the candidate sites based
on the criteria below. Typically the easiest way to do this is to follow the workflow shown in the defect
generation tutorial, and then run the ``ShakeNBreak`` ``vasp_gam`` relaxations for the ``Unperturbed`` and
``Bond_Distortion_0.0%``/``Rattled`` directories of each charge state. Alternatively, you can generate the
``vasp_gam`` relaxation input files by setting ``vasp_gam = True`` in ``DefectsSet.write_files()``.

We can then compare the energies of these trial relaxations, and remove candidates that either:

- Are high energy (~>1 eV above the lowest energy site for each charge state), and so are unlikely to form.

- Relax to the same final structure/energy as other interstitial sites (despite different initial
  positions) in each charge state, and so are unnecessary to calculate. This can happen due to interstitial
  migration within the relaxation calculation, from an unfavourable higher energy site, to a lower energy
  one. Typically if the energy from the test ``vasp_gam`` relaxations are within a couple of meV of each other,
  this is the case.

.. tip::

    As with many steps in the defect calculation workflow, these are only rough general guidelines and
    you should always critically consider the validity of these choices in the context of your specific
    system (for example, considering the charge-state dependence of the interstitial site formation
    energies here).

.. note::

    As mentioned above, by default Voronoi tessellation is used to generate the candidate interstitial
    sites in ``doped``. We have consistently found this approach to be the most robust in identifying all
    stable/low-energy interstitial sites across a wide variety of materials and chemistries. A nice
    discussion is given in
    `Kononov et al. J. Phys.: Condens. Matter 2023 <https://iopscience.iop.org/article/10.1088/1361-648X/acd3cf>`_.

    As with all aspects of the calculation workflow, interstitial site generation is
    flexible, and you can explicitly specify the interstitial sites to generate using the
    ``interstitial_coords`` (for instance, if you only want to investigate one specific known interstitial
    site, or input a list of candidate sites generated from a different algorithm), and/or customise the
    generation algorithm via ``interstitial_gen_kwargs``, both of which are input parameters for the
    ``DefectsGenerator`` class;
    see the `API documentation <https://doped.readthedocs.io/en/latest/doped.generation.html#doped.generation.DefectsGenerator>`_
    for more details.

    Charge-density based approaches for interstitial site generation can be useful in some cases and often
    output fewer candidate sites, but we have found that these are primarily suited to ionic materials (and
    with fully-ionised defect charge states) where electrostatics primarily govern the energetics. In
    many systems (particularly those with some presence of (ionic-)covalent bonding) where orbital
    hybridisation plays a role, this approach can often miss the ground-state interstitial site(s).
    ..  If you are limited with computational resources and are working with (relatively simple) ionic compound(s), this approach may be worth considering.


Difficult Structural Relaxations
--------------------------------

If defect supercell relaxations do not converge after multiple continuation calculations
(i.e. ``cp``-ing ``CONTCAR`` to ``POSCAR`` and resubmitting the job), this is likely due to small
residual forces causing the local optimisation algorithm to struggle to find a solution, an error in the
underlying calculation and/or extreme forces.

- If the calculation outputs show that the relaxation is proceeding fine, without any errors, just not
  converging to completion, then it suggests that the structure relaxation is bouncing around a narrow
  region of the potential energy surface. Here, the gradient-based geometry optimiser is
  struggling to converge.

    - Often (but not always) this indicates that the structure may be stuck around a `saddle point` or
      shallow local minimum on the potential energy surface (PES), so it's important to make sure
      that you have performed structure-searching (PES scanning) with an approach such as
      `ShakeNBreak <https://shakenbreak.readthedocs.io>`_ (``SnB``) to avoid this. You may want to try
      'rattling' the structure to break symmetry in case this is an issue, as detailed in
      `this part <https://shakenbreak.readthedocs.io/en/latest/Tips.html#bulk-phase-transformations>`_
      of the ``SnB`` docs.

    - **Alternatively (if you have already performed `SnB` structure-searching), convergence of the forces can be aided by:**
    - Switching the ionic relaxation algorithm back and forth (i.e. change :code:`IBRION` to :code:`1` or
      :code:`3` and back).
    - Reducing the ionic step width (e.g. change :code:`POTIM` to :code:`0.02` in the :code:`INCAR`)
    - Switching the electronic minimisation algorithm (e.g. change :code:`ALGO` to :code:`All`), if
      electronic convergence seems to be causing issues.
    - Tightening/reducing the electronic convergence criterion (e.g. change :code:`EDIFF` to :code:`1e-7`)

- If instead the calculation is crashing due to an error and/or extreme forces, a common culprit is the
  :code:`EDWAV` error in the output file, which can often be avoided by reducing :code:`NCORE` and/or
  :code:`KPAR`. If this doesn't fix it, switching the electronic minimisation algorithm (e.g. change
  :code:`ALGO` to :code:`All`) can sometimes help.

    - If some relaxations are still not converging after multiple continuations, you should check the
      calculation output files to see if this requires fixing. Often this may require changing a
      specific :code:`INCAR` setting, and using the updated setting(s) for any other relaxations that
      are also struggling to converge.

``ShakeNBreak``
-------------------

For tips on the ``ShakeNBreak`` part of the defect calculation workflow, please refer to the
`ShakeNBreak documentation <https://shakenbreak.readthedocs.io>`_.

Layered / Low Dimensional Materials
--------------------------------------
Layered and low-dimensional materials introduce complications for defect analysis. One point is that typically such lower-symmetry materials exhibit higher rates of energy-lowering defect reconstructions (e.g.
`4-electron negative-U centres in Sb₂Se₃ <https://doi.org/10.1103/PhysRevB.108.134102>`_), as a result of
having more complex energy landscapes.

Another is that often the application of charge correction schemes to supercell calculations with layered
materials may require some fine-tuning for converged results. To illustrate, for Sb₂Si₂Te₆ (
`a promising layered thermoelectric material <https://doi.org/10.26434/chemrxiv-2024-hm6vh>`_),
when parsing the intrinsic defects, the -3 charge antimony vacancy (``v_Sb-3``) gave this warning:

.. code-block::

        Estimated error in the Kumagai (eFNV) charge correction for defect v_Sb_-3 is 0.067 eV (i.e. which is
        greater than the `error_tolerance`: 0.050 eV). You may want to check the accuracy of the correction by
        plotting the site potential differences (using `defect_entry.get_kumagai_correction()` with `plot=True`).
        Large errors are often due to unstable or shallow defect charge states (which can't be accurately modelled
        with the supercell approach). If this error is not acceptable, you may need to use a larger supercell
        for more accurate energies.

Following the advice in the warning, we use ``defect_entry.get_kumagai_correction(plot=True)`` to plot the
site potential differences for the defect supercell (which is used to obtain the eFNV (Kumagai-Oba)
anisotropic charge correction):

.. image:: Sb2Si2Te6_v_Sb_-3_eFNV_plot.png
    :width: 400px
    :align: left

.. image:: Sb2Si2Te6_v_Sb_-3_VESTA.png
    :width: 240px
    :align: right

From the eFNV plot, we can see that there appears to be two distinct sets of site potentials, with one
curving up from ~-0.4 V to ~0.1 V, and another mostly constant set at ~0.3 V. We can understand this by
considering the structure of our defect (shown on the right), where the location of the Sb vacancy (hidden
by the projection along the plane) is circled in green – we can see the displacement of the Sb atoms on
either side.

Due to the layered structure, the charge and strain associated with the defect is mostly confined to the
defective layer, while that of the layer away from the defect mostly experiences the typical long-range
electostatic potential of the defect charge. The same behaviour can be seen for `h`-BN in the
`original eFNV paper <https://doi.org/10.1103/PhysRevB.89.195205>`_ (Figure 4d).
This means that our usual default of using the
Wigner-Seitz radius to determine the sampling region is not as good, as it's including sites in the
defective layer (circled in orange) which are causing the variance in the potential offset (ΔV) and thus
the error in the charge correction.

To fix this, we can use the optional ``defect_region_radius`` or ``excluded_indices`` parameters in
``get_kumagai_correction``, to exclude those points from the sampling. For ``defect_region_radius``, we
can just set this to 8.75 Å here to avoid those sites in the defective layer. Often it may not be so simple
to exclude the intra-layer sites in this way (depending on the supercell), and so alternatively we can use
``excluded_indices`` for more fine-grained control. As we can see in the structure image above, the `a`
lattice vector is aligned along the inter-layer direction, so we can determine the intra-layer sites using
the fractional coordinates of the defect site along `a`:

.. code-block:: python

    # get indices of sites within 0.2 fractional coordinates along a of the defect site
    sites_in_layer = [
        i for i, site in enumerate(defect_entry.defect_supercell)
        if abs(site.frac_coords[0] - defect_entry.defect_supercell_site.frac_coords[0]) < 0.2
    ]
    correction, fig =  dp.defect_dict["v_Sb-3"].get_kumagai_correction(
        excluded_indices=sites_in_layer, plot=True
    )  # note that this updates the DefectEntry.corrections value, so the updated correction
    # is used in later formation energy / concentration calculations

Below are the two resulting charge correction plots (using ``defect_region_radius`` on the left, and
``excluded_indices`` on the right):

.. image:: Sb2Si2Te6_v_Sb_-3_eFNV_plot_region_radius.png
    :width: 320px
    :align: left

.. image:: Sb2Si2Te6_v_Sb_-3_eFNV_plot_no_intralayer.png
    :width: 320px
    :align: right

Perturbed Host States
--------------------------------------

Certain point defects form shallow (hydrogen-like) donor or acceptor states, known as perturbed host
states (PHS). These states typically have wavefunctions distributed over many unit cells in real space,
requiring exceptionally large supercells or dense reciprocal space sampling to properly capture their
physics (`see this review <https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.50.797>`_).
This weak attraction of the electron/hole to the defect site corresponds to a relatively small
donor/acceptor binding energy (i.e. energetic separation of the corresponding charge transition level to
the nearby band edge), which is typically <100 meV.

Current supercell correction schemes can not accurately account for finite-size errors obtained when
calculating the energies of PHS in moderate supercells, so it is recommended to denote such shallow defects
as PHS and conclude only `qualitatively` that their transition level is located near the corresponding
band edge. An example of this is given in `Kikuchi et al. Chem. Mater. 2020
<https://doi.org/10.1021/acs.chemmater.1c00075>`_.

```{tip}
Typically, the shallow defect binding energy can be reasonably well estimated using the hydrogenic model,
similar to the `Wannier-Mott <https://en.wikipedia.org/wiki/Exciton#Wannier%E2%80%93Mott_exciton>`__
exciton model, which predicts a binding energy given by:

.. math::

   E_b = \text{13.6 eV} \times \frac{\bar{m}}{\epsilon^2}

where :math:`\bar{m}` is the harmonic mean (i.e. conductivity) effective mass of the relevant
charge-carrier (electron/hole), :math:`\epsilon` is the total dielectric constant
(:math:`\epsilon = \epsilon_{\text{ionic}} + \epsilon_{\infty}`) and 13.6 eV is the Rydberg constant (i.e.
binding energy of an electron in a hydrogen atom).
```

We employ the methodology of `Kumagai et al. <https://doi.org/10.1103/PhysRevMaterials.5.123803>`_ to
identify potential PHS through an interface with ``pydefect``.

The optional argument ``load_phs_data`` in ``DefectsParser`` (``True`` by default) controls whether to
load the projected orbitals, and in combination with ``defect_entry.get_perturbed_host_state()`` returns
additional information about the nature of the band edges, allowing defect states (and whether they are
deep or shallow (PHS)) to be automatically identified.
Furthermore, a plot for the single particle levels is returned. It is however recommended to manually
check the real-space charge density (i.e. ``PARCHG``) of the defect state to confirm the identification of
a PHS. Your ``INCAR`` file needs to include ``LORBIT > 10`` to obtain the projected orbitals and your
bulk calculation folder must contain the ``OUTCAR(.gz)`` file.

In the example below, the neutral copper vacancy in `Cu₂SiSe₃ <https://doi.org/10.1039/D3TA02429F>`_ was
determined to be a PHS. This was additionally confirmed by performing calculations in larger
supercells and plotting the charge density. Important terms include:
1) ``P-ratio``: The ratio of the summed projected orbital contributions of the defect & neighbouring sites
to the total sum of orbital contributions from all atoms to that electronic state. A value close to 1
indicates a localised state.
2) ``Occupation``: Occupation of the electronic state / orbital.
3) ``vbm has acceptor phs``/``cbm has donor phs``: Whether a PHS has been automatically identified.
Depends on how VBM-like/CBM-like the defect states are and the occupancy of the state. ``(X vs. 0.2)``
refers to the hole/electron occupancy at the band edge vs the default threshold of 0.2 for flagging as a
PHS (but you should use your own judgement of course).
4) ``Localized Orbital(s)``: Information about the localised defects states.

.. code-block:: python

    bulk = "Cu2SiSe3/bulk/vasp_std"
    defect = "Cu2SiSe3/v_Cu_0/vasp_std/"

    defect = DefectParser.from_paths(defect,bulk,dielectric,skip_corrections=True).defect_entry
    bes, fig = defect.get_perturbed_host_state()
    print(bes)  # print information about the defect state

     -- band-edge states info
    Spin-up
         Index  Energy  P-ratio  Occupation  OrbDiff  Orbitals                            K-point coords
    VBM  347    3.539   0.05     1.00        0.03     Cu-d: 0.34, Se-p: 0.36              ( 0.000,  0.000,  0.000)
    CBM  348    5.139   0.04     0.00        0.02     Se-s: 0.20, Se-p: 0.12, Si-s: 0.13  ( 0.000,  0.000,  0.000)
    vbm has acceptor phs: False (0.000 vs. 0.2)
    cbm has donor phs: False (0.000 vs. 0.2)
    ---
    Localized Orbital(s)
    Index  Energy  P-ratio  Occupation  Orbitals

    Spin-down
         Index  Energy  P-ratio  Occupation  OrbDiff  Orbitals                            K-point coords
    VBM  347    3.677   0.06     0.00        0.02     Cu-d: 0.34, Se-p: 0.36              ( 0.000,  0.000,  0.000)
    CBM  348    5.142   0.04     0.00        0.02     Se-s: 0.20, Se-p: 0.12, Si-s: 0.13  ( 0.000,  0.000,  0.000)
    vbm has acceptor phs: True (1.000 vs. 0.2)
    cbm has donor phs: False (0.000 vs. 0.2)
    ---
    Localized Orbital(s)
    Index  Energy  P-ratio  Occupation  Orbitals

.. note::

    Have any tips for users from using ``doped``? Please share it with the developers and we'll add them here!