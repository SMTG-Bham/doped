Tips & Tricks
============================

Interstitials
-------------------
As described in the `YouTube defect calculation tutorial <https://youtu.be/FWz7nm9qoNg>`_, our
recommended workflow for calculating interstitial defects is to first generate the set of
possible interstitial sites for your structure using ``DefectsGenerator`` (which uses Voronoi tessellation
to do this), and then perform Gamma-point-only relaxations (using ``vasp_gam``) for the neutral state of
each generated interstitial candidate. The ``vasp_gam`` relaxation files can be generated following the
steps shown in the
`defect generation tutorial <https://doped.readthedocs.io/en/latest/dope_workflow_example.html>`_ and
setting ``vasp_gam = True`` in ``DefectsSet.write_files()``.

We can then compare the energies of these trial neutral relaxations, and remove any candidates that
either:

- Are very high energy (>1 eV above the lowest energy site), and so are unlikely to form.

- Relax to the same final structure/energy as other interstitial sites (despite different initial
  positions), and so are unnecessary to calculate. This can happen due to interstitial migration within
  the relaxation calculation, from an unfavourable higher energy site, to a lower energy one. Typically
  if the energy from the test neutral `vasp_gam` relaxations are within a couple meV of eachother, this
  is the case.

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
      that you have performed structure-searching (PES scanning) with
      `ShakeNBreak <https://shakenbreak.readthedocs.io>`_ (``SnB``) to avoid this. You may want to try
      'rattling' the structure to break symmetry in case this is an issue, as detailed in
      `this part <https://shakenbreak.readthedocs.io/en/latest/Tips.html#bulk-phase-transformations>`_
      of the ``SnB`` docs.

    - **Alternatively (if you have already performed `SnB` structure-seaerching), convergence of the forces can be aided by:**
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

Have any tips for users from using ``doped``? Please share it with the developers and we'll add them here!