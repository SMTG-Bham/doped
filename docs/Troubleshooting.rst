.. _troubleshooting:

Troubleshooting & Support
=========================

``doped``/``pymatgen`` Errors
-----------------------------
For most error cases, ``doped`` has been designed to try and give informative error messages about why
the functions are failing.
In the majority of cases, if you encounter an error using ``doped`` which does not have a clear error
message about the origin of the problem, it is likely to be an issue with your version of ``pymatgen``
(and/or ``doped``), and may be fixed by doing:

.. code:: bash

  pip install pymatgen pymatgen-analysis-defects monty --upgrade
  pip install doped --upgrade

If this does not solve your issue, please check the specific cases noted below.
The next recommended step is to search through the ``doped``
`GitHub Issues <https://github.com/SMTG-Bham/doped/issues>`_ (use the GitHub search bar on the top
right) to see if your issue/question has been asked before. If your problem is still not solved, then
please contact the developers through the
`GitHub Issues <https://github.com/SMTG-Bham/doped/issues>`_ page.

Parsing Errors
--------------
If errors occur during parsing of defect calculations, ``doped`` will try to informatively warn you about
the origin of the parsing failure (e.g. ``Parsing failed for [...] with the same error: ...``).
Depending on what the error is, this error message on its own may not be very helpful. In these cases, it's
worth trying to parse one or two of these failing defect calculations individually, using
``DefectParser.from_paths(defect_path="...", bulk_path="...", ...)``, which should give a more verbose
error traceback.

.. note::

    "``ParseError``", ``ElementTree``/"no element found" or other XML-related errors  are related to
    issues in parsing ``vasprun.xml(.gz)`` files. In most cases, these error messages are indicating a
    corrupted/incomplete ``vasprun.xml(.gz)`` file, for which the solution is to re-run the VASP
    calculation to obtain the appropriate output.

Mis-matching Bulk and Defect Supercells
----------------------------------------
When parsing defect calculations with ``doped``, if you provide bulk and defect supercells which do not
match, you will see the following warning:

.. code::

    Detected atoms far from the defect site (>X Å) with major displacements (>0.5 Å) in the defect
    supercell. This likely indicates a mismatch between the bulk and defect supercell definitions or an
    unconverged supercell size, both of which could cause errors in parsing. The mean displacement of the
    following species, at sites far from the determined defect position, is >0.5 Å: ...

This can sometimes happen due to the use of a bulk supercell which does not match the atomic positions of
the defect supercell, but is symmetry-equivalent by a translation and/or rotation. This causes issues for
determining the defect position in the supercell, and for calculating charge corrections which rely on
differences in electrostatic potential between the bulk and defect supercells. ``doped`` will never output
mis-matching bulk and defect supercells, but this can occur from accidental combination of outputs from
old and newer versions, or separate manual calculations of bulk supercells etc.

The easiest solution is to generate the bulk supercell which corresponds to the defect supercell
definitions:

.. code:: python

    from pymatgen.core.structure import Structure
    from doped.utils.configurations import orient_s2_like_s1

    # Load the bulk and defect supercells
    defect_supercell = Structure.from_file("...")
    bulk_supercell = Structure.from_file("...")  # which mis-matches the defect supercell

    # orient the bulk supercell to match the defect supercell:
    # for this, we need to 'reverse' the defect formation in the defect supercell, to get
    # a supercell with the bulk composition that can then be used as a rough template.
    # in this example case we add a site because the defect supercell is an oxygen vacancy,
    # but for interstitials you would remove a site (with Structure.remove(...)) and for
    # substitutions you would replace a site (with Structure.replace(...)).
    defect_supercell_w_bulk_comp = defect_supercell.copy()
    defect_supercell_w_bulk_comp.append("O", [0.5, 0.5, 0.5])  # add element to remove vacancy
    # defect frac coords are given in the POSCAR comment

    oriented_bulk_supercell = orient_s2_like_s1(defect_supercell_w_bulk_comp, bulk_supercell)
    oriented_bulk_supercell.to(fmt="POSCAR", filename="oriented_bulk_POSCAR")

With this re-generated matching bulk supercell, we just need to run the single-point bulk calculation, and
use this matching-supercell calculation for defect parsing.

If for some reason you have different supercell definitions for different sets of defect calculations, you
can use different bulk supercells (which match the corresponding set of defect supercells) and combine them
using something like:

.. code:: python

    from doped.analysis import DefectsParser
    from doped.thermodynamics import DefectThermodynamics

    dp_1 = DefectsParser("Defects_Calcs_Supercell_1", bulk_path="Bulk_Supercell_1", dielectric=dielectric)
    dp_2 = DefectsParser("Defects_Calcs_Supercell_2", bulk_path="Bulk_Supercell_2", dielectric=dielectric)

    thermo = DefectThermodynamics([*dp_1.defect_entries.values(), *dp_2.defect_entries.values()], chempots...)


``numpy`` Errors
-------------------
A previous known issue with ``numpy``/``pymatgen`` is that it could give an error similar to this:

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


``spglib`` Errors/Warnings
--------------------------
A known issue with ``spglib`` is that it can give unnecessary errors or warnings similar to:

.. code:: python

  spglib: ssm_get_exact_positions failed (attempt=0).
  spglib: No point group was found (line 405, ...).
  ...
  spglib: ssm_get_exact_positions failed (attempt=4).
  spglib: get_bravais_exact_positions_and_lattice failed
  spglib: ref_get_exact_structure_and_symmetry failed.

Typically this can be fixed by updating to ``spglib>=2.5`` with `pip install --upgrade spglib``.
.. see doped_spglib_warnings.ipynb

``ShakeNBreak``
-------------------

For issues relating to the ``ShakeNBreak`` part of the defect calculation workflow, please refer to the
`ShakeNBreak documentation <https://shakenbreak.readthedocs.io>`_.

Installation
------------

For any issues relating to installation, please see the `Installation`_ page.


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

.. code-block::  none

    RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.
        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:
            if __name__ == '__main__':
                freeze_support()
                ...
        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
        To fix this issue, refer to the "Safe importing of main module"
        section in https://docs.python.org/3/library/multiprocessing.html

.. _tutorials: https://doped.readthedocs.io/en/latest/Tutorials.html

.. NOTE::
    If you run into any issues using ``doped`` that aren't addressed above, please contact the developers
    through the ``GitHub`` `Issues <https://github.com/SMTG-Bham/doped/issues>`_ page.

.. _Installation: https://doped.readthedocs.io/en/latest/Installation.html