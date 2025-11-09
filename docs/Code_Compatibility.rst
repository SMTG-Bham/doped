Code Compatibility
========================

Python Package Compatibility
----------------------------
:code:`doped` is built to natively function using (and being compatible with) the most recent version of
:code:`pymatgen`. If you are receiving :code:`pymatgen`-related errors when using
:code:`doped`, you may need to update :code:`pymatgen` and/or :code:`doped`, which can be done with:

.. code:: bash

   pip install --upgrade pymatgen doped

.. note::
  If you run into any errors when using ``doped``, please check the :ref:`Troubleshooting` page, where any 
  known issues with codes (such as ``numpy``, ``scipy`` or ``spglib``) are detailed.

Energy Calculator (DFT/ML) Compatibility
----------------------------------------
The vast majority of the code in :code:`doped` is agnostic to the underlying energy calculator / electronic
structure (i.e. DFT/ML) code used to calculate the raw energies of defect supercells. However, as
demonstrated in the tutorials, direct I/O support is currently provided for the ``VASP`` DFT code, while
structure files for essentially all DFT/ML codes can be easily generated using the
:meth:`~pymatgen.core.structure.IStructure.to()` method for ``pymatgen`` 
:class:`~pymatgen.core.structure.Structure`\s or
`ase I/O methods <https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.write>`__, with the
:class:`~pymatgen.core.structure.Structure` objects used for crystal structures in ``doped``.

Direct I/O capabilities for other codes is a goal (such as ``Quantum Espresso``, ``CP2K`` and/or
``FHI-aims``), accompanied by an update publication, so please get in touch with the developers if you
would be interested in contributing to this effort! Input file generation and structure/energy parsing
should be straightforward with the ``doped``/``pymatgen``/``ase`` APIs, while finite-size charge
corrections would be the main challenge for parsing & computing.

Please let us know if you have any issues with compatibility!
