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
`Issues <https://github.com/SMTG-Bham/doped/issues>`_ page, or by email.


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

Errors with ``Python`` Scripts
------------------------------
The recommended usage of ``doped`` is through interactive python sessions, such as with Jupyter notebooks,
``IPython`` or an IDE (e.g. ``PyCharm`` or ``VSCode``), as shown in the ``doped`` `tutorials`_.
However, it is possible to also use ``doped`` through ``Python`` scripts if preferred.
Due to the use of the ``multiprocessing`` module in ``doped.generation``, ``doped.vasp`` and
``doped.analysis``, you need to use the proper syntax for running ``Python`` scripts, with
``if __name__ == '__main__':...``

A simple example script of generating the intrinsic defects and writing the VASP input files (all with
default settings – in reality you likely need to customise some options!) would be:

.. code:: python

    from pymatgen.core.structure import Structure
    from doped import generation, vasp

    def generate_and_write_vasp_files():
        primitive_struct = Structure.from_file("prim_POSCAR")
        # generate defects:
        defect_gen = generation.DefectsGenerator(primitive_struct)
        # generate VASP input files:
        defects_set = vasp.DefectsSet(defect_gen)
        defects_set.write_files()

    if __name__ == '__main__':
        generate_and_write_vasp_files()

If you do not use the ``if __name__ == '__main__':...`` syntax, you may encounter this error:

.. code:: python

    RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.
        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:
            if __name__ == ‘__main__‘:
                freeze_support()
                ...
        The “freeze_support()” line can be omitted if the program
        is not going to be frozen to produce an executable.
        To fix this issue, refer to the “Safe importing of main module”
        section in https://docs.python.org/3/library/multiprocessing.html

.. _tutorials: https://doped.readthedocs.io/en/latest/Tutorials.html

.. NOTE::
    If you run into any issues using ``doped``, please contact the developers through the ``GitHub``
    `Issues <https://github.com/SMTG-Bham/doped/issues>`_ page, or by email.
