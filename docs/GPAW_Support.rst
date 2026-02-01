GPAW Support
====================================

``doped`` provides comprehensive support for defect calculations using `GPAW <https://wiki.fysik.dtu.dk/gpaw/>`_, 
a Density Functional Theory (DFT) Python code based on the projector-augmented wave (PAW) method. 
This support includes automated input generation, parsing of calculation results, and 
integration with the full ``doped`` defect analysis workflow.

Installation & Requirements
---------------------------

To use the GPAW interface, you must have GPAW installed in your Python environment:

.. code-block:: bash

    pip install gpaw

For the Kumagai (eFNV) charge correction, ``pydefect`` is also required:

.. code-block:: bash

    pip install pydefect

Workflow Overview
-----------------

The workflow for GPAW defect calculations follows the standard ``doped`` logic:

1.  **Generation**: Generate defect structures using ``DefectsGenerator``.
2.  **Input Preparation**: Write GPAW Python scripts and structure files using ``GPAWDefectRelaxSet``.
3.  **Execution**: Run the calculations using GPAW (typically via ``mpirun``).
4.  **Parsing**: Parse the results (``.gpw`` files) using ``GPAWDefectsParser``.
5.  **Analysis**: Perform thermodynamic analysis and plotting.

Input Generation
----------------

The ``GPAWDefectRelaxSet`` class is used to generate the necessary files for a GPAW relaxation. 
It produces a ``relax.py`` script and a ``structure.cif`` file.

.. code-block:: python

    from doped.gpaw import GPAWDefectRelaxSet
    from pymatgen.core.structure import Structure

    # Load your supercell structure
    structure = Structure.from_file("POSCAR")

    # Define GPAW settings
    gpaw_settings = {
        "mode": {"name": "pw", "ecut": 400},
        "xc": "PBE",
        "kpts": {"size": (2, 2, 2), "gamma": True},
    }

    # Initialize the relax set for a +1 charge state
    relax_set = GPAWDefectRelaxSet(structure, charge_state=1, gpaw_settings=gpaw_settings)

    # Write files to a directory
    relax_set.write_input("calculation_folder")

Parsing Results
---------------

Once calculations are complete, ``doped`` can parse the resulting ``relaxed.gpw`` files. 
The ``GPAWDefectsParser`` can handle multiple defect folders at once.

.. code-block:: python

    from doped.gpaw import GPAWDefectsParser

    # Initialize the parser
    # output_path: directory containing defect folders
    # bulk_path: directory containing the bulk reference calculation
    parser = GPAWDefectsParser(
        output_path=".", 
        bulk_path="calculation_bulk",
        dielectric=10.0  # Required for charge corrections
    )

    # Parse all defects
    defect_dict = parser.parse_all()

Finite-Size Corrections
-----------------------

GPAW calculations of charged defects require finite-size corrections to account for periodic 
image interactions. ``doped`` supports the **Kumagai (eFNV)** correction for GPAW.

Anisotropic Systems (2D/1D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For anisotropic systems like 2D materials (e.g., graphene, MoS2), the default sampling radius 
calculation in standard tools often fails by setting a radius that encompasses the entire cell. 
``doped`` implements an improved radius calculation:

- It automatically determines the optimal ``defect_region_radius`` based on the inscribed 
  sphere of the supercell (half the shortest distance between parallel planes).
- It includes safety checks to prevent errors when the sampling region is small or empty.

Detailed API
------------

For more specific information on classes and functions, see the :ref:`doped.gpaw module` documentation.

Example Script
--------------

An end-to-end example of generating and parsing GPAW defects can be found in the 
``examples/Graphene_with_GPAW`` directory of the ``doped`` repository.
