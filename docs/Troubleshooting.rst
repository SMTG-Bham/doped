.. _troubleshooting:

Troubleshooting
================

``doped``/``pymatgen`` Errors
-----------------------------

For most error cases, ``doped`` has been designed to try and give informative error messages about why
the functions are failing.
In the majority of cases, if you encounter an error using ``doped`` which does not have a clear error
message about the origin of the problem, it is likely to be an issue with your version of ``pymatgen``
(and/or ``doped``), and may be fixed by doing:

.. code:: bash

  pip install pymatgen --upgrade
  pip install doped --upgrade

If this does not solve your issue, please contact the developers through the ``GitHub``
`Issues <https://github.com/SMTG-UCL/doped/issues>`_ page, or by email.


``numpy`` Errors
-------------------
- A previous known issue with ``numpy``/``pymatgen`` is that it could give an error similar to this:

  .. code:: python

      ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject

  This should be avoided with current versions of ``doped``, due to the package installation
  requirements (handled automatically by ``pip``), but depending on your ``python`` environment and
  previously-installed packages, it could possibly still arise. It occurs due to a recent change in the
  ``numpy`` C API in version ``1.20.0``, see
  `here <https://stackoverflow.com/questions/66060487/valueerror-numpy-ndarray-size-changed-may-indicate-binary-incompatibility-exp>`_
  for details.
  It should be fixed by reinstalling ``numpy`` and ``pymatgen`` (so that they play nice together), so
  that it is rebuilt with the new ``numpy`` C API:

  .. code:: bash

      pip install --force --no-cache-dir numpy==1.23
      pip uninstall pymatgen
      pip install pymatgen


``ShakeNBreak``
-------------------

For issues relating to the ``ShakeNBreak`` part of the defect calculation workflow, please refer to the
`ShakeNBreak documentation <https://shakenbreak.readthedocs.io>`_.


.. NOTE::
    If you run into any issues using ``doped``, please contact the developers through the ``GitHub``
    `Issues <https://github.com/SMTG-UCL/doped/issues>`_ page, or by email.
