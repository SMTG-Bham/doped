doped.GPAW_sets
=================================================

The ``yaml`` file below defines the default settings used by ``doped`` for ``GPAW`` defect and bulk supercell
calculations, and is located in the `GPAW_sets`_ folder of the ``doped`` GitHub repository. It is loaded as the
module-level dictionary :data:`~doped.io.gpaw.inputs.default_gpaw_settings`, and any of its settings can be
overridden with the ``gpaw_settings`` argument of :class:`~doped.io.gpaw.inputs.DefectsSet` and
:class:`~doped.io.gpaw.inputs.DefectDictSet` (see :doc:`GPAW_Support`).

.. _GPAW_DefectSet.yaml:

``GPAW_DefectSet.yaml``
-------------------------------------------------
Default settings for defect supercell relaxations and bulk supercell single-point calculations.

.. literalinclude:: ../doped/io/gpaw/GPAW_sets/GPAW_DefectSet.yaml
   :language: yaml

.. _GPAW_sets:
   https://github.com/SMTG-Bham/doped/tree/main/doped/io/gpaw/GPAW_sets
