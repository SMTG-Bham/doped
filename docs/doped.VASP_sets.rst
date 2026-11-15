doped.VASP_sets
=================================================

The ``yaml`` files below define the default ``INCAR``, ``KPOINTS`` and ``POTCAR`` settings used by ``doped`` for
different ``VASP`` calculation types, and are located in the `VASP_sets`_ folder of the ``doped`` GitHub repository.
Each is loaded as the module-level dictionary in :mod:`~doped.io.vasp.inputs` linked below.

.. _RelaxSet.yaml:

``VASP_RelaxSet.yaml``
-------------------------------------------------
Base settings for geometry relaxations; loaded as :data:`~doped.io.vasp.inputs.default_relax_set`.

.. literalinclude:: ../doped/io/vasp/VASP_sets/VASP_RelaxSet.yaml
   :language: yaml

.. _DefectSet.yaml:

``VASP_DefectSet.yaml``
-------------------------------------------------
``INCAR`` settings specific to defect supercell calculations, applied on top of ``VASP_RelaxSet.yaml``;
loaded as :data:`~doped.io.vasp.inputs.default_defect_set` (and merged in :data:`~doped.io.vasp.inputs.default_defect_relax_set`).

.. literalinclude:: ../doped/io/vasp/VASP_sets/VASP_DefectSet.yaml
   :language: yaml

.. _HSESet.yaml:

``VASP_HSESet.yaml``
-------------------------------------------------
``INCAR`` settings for hybrid DFT calculations (HSE06 by default); loaded as :data:`~doped.io.vasp.inputs.default_HSE_set`.

.. literalinclude:: ../doped/io/vasp/VASP_sets/VASP_HSESet.yaml
   :language: yaml

.. _SinglePointSet.yaml:

``VASP_SinglePointSet.yaml``
-------------------------------------------------
``INCAR`` setting overrides for single-point (static) calculations, applied on top of the relaxation
settings; loaded as :data:`~doped.io.vasp.inputs.singlepoint_incar_settings`.

.. literalinclude:: ../doped/io/vasp/VASP_sets/VASP_SinglePointSet.yaml
   :language: yaml

.. _ConvergenceSet.yaml:

``VASP_ConvergenceSet.yaml``
-------------------------------------------------
``INCAR`` settings for the GGA DFT ``k``-point convergence calculations of competing phases; loaded as
:data:`~doped.io.vasp.inputs.convergence_set`.

.. literalinclude:: ../doped/io/vasp/VASP_sets/VASP_ConvergenceSet.yaml
   :language: yaml

.. _PotcarSet.yaml:

``VASP_PotcarSet.yaml``
-------------------------------------------------
Default ``POTCAR`` functional, and ``POTCAR`` symbol for each element; the latter is loaded as
:data:`~doped.io.vasp.inputs.default_potcar_dict`.

.. literalinclude:: ../doped/io/vasp/VASP_sets/VASP_PotcarSet.yaml
   :language: yaml

.. _VASP_sets:
   https://github.com/SMTG-Bham/doped/tree/main/doped/io/vasp/VASP_sets
