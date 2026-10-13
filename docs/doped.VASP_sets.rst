doped.VASP_sets
=================================================

The ``yaml`` files below define the default ``INCAR``, ``KPOINTS`` and ``POTCAR`` settings used by ``doped`` for
different ``VASP`` calculation types, and are located in the `VASP_sets`_ folder of the ``doped`` GitHub repository.
Each is loaded as the module-level dictionary in :mod:`~doped.vasp` linked below.

.. _RelaxSet.yaml:

``RelaxSet.yaml``
-------------------------------------------------
Base settings for geometry relaxations; loaded as :data:`~doped.vasp.default_relax_set`.

.. literalinclude:: ../doped/VASP_sets/RelaxSet.yaml
   :language: yaml

.. _DefectSet.yaml:

``DefectSet.yaml``
-------------------------------------------------
``INCAR`` settings specific to defect supercell calculations, applied on top of ``RelaxSet.yaml``;
loaded as :data:`~doped.vasp.default_defect_set` (and merged in :data:`~doped.vasp.default_defect_relax_set`).

.. literalinclude:: ../doped/VASP_sets/DefectSet.yaml
   :language: yaml

.. _HSESet.yaml:

``HSESet.yaml``
-------------------------------------------------
``INCAR`` settings for hybrid DFT calculations (HSE06 by default); loaded as :data:`~doped.vasp.default_HSE_set`.

.. literalinclude:: ../doped/VASP_sets/HSESet.yaml
   :language: yaml

.. _SinglePointSet.yaml:

``SinglePointSet.yaml``
-------------------------------------------------
``INCAR`` setting overrides for single-point (static) calculations, applied on top of the relaxation
settings; loaded as :data:`~doped.vasp.singlepoint_incar_settings`.

.. literalinclude:: ../doped/VASP_sets/SinglePointSet.yaml
   :language: yaml

.. _ConvergenceSet.yaml:

``ConvergenceSet.yaml``
-------------------------------------------------
``INCAR`` settings for the GGA DFT ``k``-point convergence calculations of competing phases; loaded as
:data:`~doped.chemical_potentials.convergence_set`.

.. literalinclude:: ../doped/VASP_sets/ConvergenceSet.yaml
   :language: yaml

.. _PotcarSet.yaml:

``PotcarSet.yaml``
-------------------------------------------------
Default ``POTCAR`` functional, and ``POTCAR`` symbol for each element; the latter is loaded as
:data:`~doped.vasp.default_potcar_dict`.

.. literalinclude:: ../doped/VASP_sets/PotcarSet.yaml
   :language: yaml

.. _VASP_sets:
   https://github.com/SMTG-Bham/doped/tree/main/doped/VASP_sets
