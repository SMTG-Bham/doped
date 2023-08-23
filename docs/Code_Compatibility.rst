Code Compatibility
========================

:code:`doped` is built to natively function using (and being compatible with) the most recent version of
:code:`pymatgen`. If you are receiving :code:`pymatgen`-related errors when using
:code:`doped`, you may need to update :code:`pymatgen` and/or :code:`doped`, which can be done with:

.. code:: bash

   pip install --upgrade pymatgen doped

In particular, there were some major breaking changes in the underlying ``pymatgen`` defect functions in
July 2022, such that major refactoring was undertaken to make the code compatible with
``pymatgen > 2022.7.25`` (and by consequence is no longer compatible with older versions of ``pymatgen``
< ``2022.7.25``).

.. note::
  If you run into this ``numpy``-related error, please see the :ref:`Troubleshooting <troubleshooting>`
  section on the docs tips page:

  .. code:: python

      ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject

Please let us know if you have any issues with compatibility!
