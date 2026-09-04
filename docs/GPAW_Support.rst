.. _gpaw-support:

GPAW Support
============

``doped`` provides an interface for generating and parsing defect calculations
with `GPAW <https://gpaw.readthedocs.io/>`__, alongside the existing defect
generation, correction and thermodynamic-analysis tools. The interface supports
fixed-cell ionic relaxations and static single-point calculations.

Installation
------------

Install GPAW and its PAW datasets following the
`GPAW installation guide <https://gpaw.readthedocs.io/install.html>`__. A typical
``pip`` installation is:

.. code-block:: bash

   pip install gpaw
   gpaw install-data

``pydefect``, which is used by ``doped`` for the Kumagai (eFNV) correction, is
already a core ``doped`` dependency.

Workflow
--------

The GPAW workflow follows the usual ``doped`` sequence:

#. Generate defect supercells with
   :class:`~doped.generation.DefectsGenerator`.
#. Write ``structure.cif`` and GPAW Python inputs with
   :class:`~doped.gpaw.GPAWDefectRelaxSet`.
#. Run the GPAW calculations locally or through a scheduler.
#. Parse the bulk and defect ``.gpw``/``.gpw.gz`` restart files with
   :class:`~doped.gpaw.GPAWDefectsParser`.
#. Analyse the resulting :class:`~doped.core.DefectEntry` objects with
   :class:`~doped.thermodynamics.DefectThermodynamics`.

Input generation
----------------

The calculator settings are supplied as a dictionary. Settings consumed by the
input generator, such as ``optimizer``, ``fmax`` and ``legacy_gpaw``, are not
passed on to the GPAW calculator.

.. code-block:: python

   from pymatgen.core import Structure

   from doped.gpaw import GPAWDefectRelaxSet

   structure = Structure.from_file("POSCAR")
   gpaw_settings = {
       "mode": {"name": "pw", "ecut": 400},
       "xc": "PBE",
       "kpts": {"size": (2, 2, 2), "gamma": True},
       "optimizer": "FIRE",
       "fmax": 0.05,
   }

   relax_set = GPAWDefectRelaxSet(
       structure,
       charge_state=1,
       gpaw_settings=gpaw_settings,
   )
   relax_set.write_input("v_Mg_+1")

This writes ``v_Mg_+1/structure.cif`` and ``v_Mg_+1/relax.py``. Static
single-point inputs can be generated with ``calculation_type="singlepoint"``:

.. code-block:: python

   singlepoint_set = GPAWDefectRelaxSet(
       structure,
       charge_state=1,
       gpaw_settings=gpaw_settings,
       calculation_type="singlepoint",
   )
   singlepoint_set.write_input("v_Mg_+1_singlepoint")

Site-resolved initial magnetic moments can be set through
``gpaw_settings={"initial_magnetic_moments": [...]}``. For example, an isolated
triplet O2 reference can be initialised with ``[1.0, 1.0]``.

Running the calculations
------------------------

The generated scripts can be run serially with ``python relax.py`` or in
parallel, for example:

.. code-block:: bash

   mpirun -np 8 gpaw python relax.py

Converge the plane-wave cutoff, k-point sampling, supercell size, smearing and
relaxation settings for the material being studied. The example settings are
chosen to demonstrate the workflow and are not universal production settings.

Parsing and charge corrections
------------------------------

``GPAWDefectsParser`` parses completed calculations during initialisation. It
automatically finds recognised GPAW restart files inside each calculation
directory and stores the parsed entries in ``defect_dict``:

.. code-block:: python

   from doped.gpaw import GPAWDefectsParser

   parser = GPAWDefectsParser(
       output_path=".",
       bulk_path="bulk",
       dielectric=8.8963,
   )
   defect_dict = parser.defect_dict

When a dielectric constant is supplied, the Kumagai (eFNV) finite-size charge
correction is applied by default. The Freysoldt (FNV) correction is also
supported using the parsed planar-averaged potentials. To compare it for a
particular entry, first remove the existing eFNV correction to avoid counting
both corrections:

.. code-block:: python

   entry = defect_dict["v_Mg_+1"]
   entry.corrections.pop("kumagai_charge_correction", None)
   entry.corrections_metadata.pop("kumagai_charge_correction", None)
   entry.get_freysoldt_correction()

Chemical potentials and formation energies
------------------------------------------

Competing-phase calculations must use energies that are consistent with the
GPAW settings used for the bulk and defects. Individual outputs can be converted
to ``pymatgen`` computed entries and used to determine the chemical-potential
limits:

.. code-block:: python

   from doped.chemical_potentials import get_doped_chempots_from_entries
   from doped.gpaw import GPAWParser
   from doped.thermodynamics import DefectThermodynamics

   bulk_parser = GPAWParser("bulk")
   bulk_entry = bulk_parser.get_computed_structure_entry()
   bulk_parser.close()

   # phase_entries should contain consistently calculated elemental and
   # competing-phase entries.
   chempots = get_doped_chempots_from_entries(
       [bulk_entry, *phase_entries],
       bulk_entry,
   )
   defect_thermodynamics = DefectThermodynamics(defect_dict, chempots=chempots)
   defect_thermodynamics.plot(limit="Mg-rich")
   defect_thermodynamics.plot(limit="O-rich")

The complete MgO example prepares Mg structures obtained from the
:class:`~doped.chemical_potentials.CompetingPhases` workflow and an isolated
triplet O2 reference, then generates Mg-rich and O-rich formation-energy
diagrams.

Anisotropic and two-dimensional systems
---------------------------------------

An anisotropic dielectric tensor should be supplied for anisotropic systems.
For example, the graphene regression test uses
``numpy.diag([1e6, 1e6, 1.0])`` to represent metallic in-plane screening and a
vacuum-like out-of-plane response. If the default Wigner-Seitz defect region
leaves no atomic sampling sites, ``doped`` reduces the sampling radius and
issues a warning.

This fallback prevents an empty sampling set, but it does not replace a
finite-size correction specifically derived for two-dimensional boundary
conditions. Quantitative 2D results should therefore be checked against an
appropriate specialised correction scheme and supercell convergence tests.

Examples and API
----------------

- ``examples/MgO_with_GPAW``: three-dimensional MgO workflow, competing-phase
  references, chemical potentials and formation-energy diagrams.
- ``examples/Graphene_with_GPAW``: anisotropic two-dimensional parsing example.
- :doc:`doped.gpaw`: complete GPAW API reference.
