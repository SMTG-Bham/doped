Code Compatibility
========================

Python Package Compatibility
----------------------------
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
  If you run into any errors when using ``doped``, please check the
  :ref:`Troubleshooting <troubleshooting>` docs page, where any known issues with codes (such as ``numpy``,
  ``scipy`` or ``spglib``) are detailed.

Energy Calculator (DFT/ML) Compatibility
----------------------------------------
The vast majority of the code in :code:`doped` is agnostic to the underlying energy calculator / electronic
structure (i.e. DFT/ML) code used to calculate the raw energies of defect supercells. However, as
demonstrated in the tutorials, direct I/O support is currently provided for the ``VASP`` DFT code, while
structure files for essentially all DFT/ML codes can be easily generated using the
`pymatgen <https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.IStructure.to>`__ or
`ase <https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.write>`__ I/O methods, with the ``pymatgen``
``Structure`` objects used for crystal structures in ``doped``.

Direct I/O capabilities for other codes is a goal (such as ``Quantum Espresso``, ``CP2K`` and/or
``FHI-aims``), accompanied by an update publication, so please get in touch with the developers if you
would be interested in contributing to this effort! Input file generation and structure/energy parsing
should be straightforward with the ``doped``/``pymatgen``/``ase`` APIs, while finite-size charge
corrections would be the main challenge for parsing & computing.

Please let us know if you have any issues with compatibility!
