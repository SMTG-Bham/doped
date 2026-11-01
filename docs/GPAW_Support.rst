.. _gpaw-support:

GPAW Support
============

``doped`` provides an interface for generating and parsing defect calculations
with `GPAW <https://gpaw.readthedocs.io/>`__, alongside the existing defect
generation, correction and thermodynamic-analysis tools.

Installation
------------

Install GPAW and its PAW datasets following the
`GPAW installation guide <https://gpaw.readthedocs.io/install.html>`__. A typical
``pip`` installation is:

.. code-block:: bash

   pip install doped[gpaw]
   gpaw install-data

``GPAW`` 25.7.0 or later is required.

.. note::

    GPAW support is **experimental**. See the tracking issue for the features
    that are not yet wired up for GPAW outputs.

Workflow
--------

The GPAW workflow follows the usual ``doped`` sequence:

#. Generate defect supercells with |DefectsGenerator|.
#. Write ``structure.cif`` and GPAW Python inputs with
   :class:`~doped.io.gpaw.inputs.DefectsSet`.
#. Run the GPAW calculations locally or through a scheduler.
#. Parse the bulk and defect ``.gpw`` restart files with |DefectsParser| (with
   ``calculator="gpaw"``).
#. Analyse the resulting |DefectEntry| objects with |DefectThermodynamics|.

Input generation
----------------

The calculator settings are supplied as a dictionary. Settings consumed by the
input generator, such as ``optimizer`` and ``fmax``, are not passed on to the
GPAW calculator. :class:`~doped.io.gpaw.inputs.DefectsSet` writes the inputs
for all defects from a |DefectsGenerator| (or a dictionary/list of
|DefectEntry| objects), with ``structure.cif`` and a ``relax.py`` GPAW script in
a ``<defect name>/`` folder for each, and a static ``singlepoint.py`` reference
calculation for the bulk supercell in ``{formula}_bulk/`` (e.g. ``MgO_bulk/``):

.. code-block:: python

   from pymatgen.core import Structure

   from doped.generation import DefectsGenerator
   from doped.io.gpaw import DefectsSet

   defect_gen = DefectsGenerator(Structure.from_file("MgO_POSCAR"))
   gpaw_settings = {
       "mode": {"name": "pw", "ecut": 400},
       "xc": "PBE",
       "kpts": {"size": (2, 2, 2), "gamma": True},
       "optimizer": "FIRE",
       "fmax": 0.05,
   }

   defects_set = DefectsSet(defect_gen, gpaw_settings=gpaw_settings)
   defects_set.write_files()

Inputs for an individual supercell |Structure| (e.g.
``DefectEntry.defect_supercell``) can be generated with
:class:`~doped.io.gpaw.inputs.DefectDictSet`:

.. code-block:: python

   from doped.io.gpaw import DefectDictSet

   structure = Structure.from_file("POSCAR")
   relax_set = DefectDictSet(
       structure,
       charge_state=1,
       gpaw_settings=gpaw_settings,
   )
   relax_set.write_input("v_Mg_+1")

This writes ``v_Mg_+1/structure.cif`` and ``v_Mg_+1/relax.py``. Static
single-point inputs can be generated with ``calculation_type="singlepoint"``:

.. code-block:: python

   singlepoint_set = DefectDictSet(
       structure,
       charge_state=1,
       gpaw_settings=gpaw_settings,
       calculation_type="singlepoint",
   )
   singlepoint_set.write_input("v_Mg_+1_singlepoint")

Initial magnetic moments can be set per element, with
``gpaw_settings={"initial_magnetic_moments": {"Fe": 4.0}}`` (unlisted elements
are set to zero), or per site with a list of one value per atom (e.g.
``[1.0, 1.0]`` for an isolated triplet O2 reference). ``DefectsSet`` requires
the per-element form, as defect and bulk supercells differ in composition.

To specify the basis setting (e.g. for LCAO calculations), supply the basis as a
GPAW calculator setting, e.g. ``gpaw_settings={"mode": {"name": "lcao"}, "basis": "dzp"}``.

Running the calculations
------------------------

The generated scripts can be run serially with ``python relax.py`` or in
parallel, for example:

.. code-block:: bash

   mpirun -np 8 gpaw python relax.py

If a relaxation is interrupted (e.g. by the job time limit), rerunning
``relax.py`` continues from the last step in ``relax.traj``. Delete
``relax.traj`` to restart from ``structure.cif``.

Converge the plane-wave cutoff, k-point sampling, supercell size, smearing and
relaxation settings for the material being studied. The example settings are
chosen to demonstrate the workflow and are not universal production settings.

Parsing and charge corrections
------------------------------

GPAW calculations are parsed with ``doped``'s usual |DefectsParser|, by setting
``calculator="gpaw"``. It finds recognised GPAW restart files inside each
calculation directory and stores the parsed entries in ``defect_dict``:

.. code-block:: python

   from doped.parsing import DefectsParser

   parser = DefectsParser(
       output_path=".",
       bulk_path="MgO_bulk",
       dielectric=8.8963,
       calculator="gpaw",
   )
   defect_dict = parser.defect_dict

When a dielectric constant is supplied, the Kumagai (eFNV) finite-size charge
correction is applied by default. The Freysoldt (FNV) correction is also
supported. To compare it for a particular entry, first remove the existing eFNV
correction to avoid counting both corrections, then pass the bulk and defect
calculation directories, from which the planar-averaged potentials are loaded
(they are not kept in ``calculation_metadata`` when the eFNV correction has
been applied):

.. code-block:: python

   entry = defect_dict["v_Mg_+1"]
   entry.corrections.pop("kumagai_charge_correction", None)
   entry.corrections_metadata.pop("kumagai_charge_correction", None)
   entry.get_freysoldt_correction(
       defect_planar_averaged_potentials="v_Mg_+1",  # the calculation directories
       bulk_planar_averaged_potentials="MgO_bulk",
   )

Chemical potentials and formation energies
------------------------------------------

GPAW competing-phase outputs can be parsed directly with
``CompetingPhasesAnalyzer(..., calculator="gpaw")``:

.. code-block:: python

   from doped.chemical_potentials import CompetingPhasesAnalyzer
   from doped.thermodynamics import DefectThermodynamics

   phase_analyzer = CompetingPhasesAnalyzer(
       "MgO",
       entries="CompetingPhases_GPAW",
       calculator="gpaw",
   )
   defect_thermodynamics = DefectThermodynamics(defect_dict, chempots=phase_analyzer.chempots)
   defect_thermodynamics.plot(limit="Mg-rich")
   defect_thermodynamics.plot(limit="O-rich")

The parser reads the stored structures, total energies and GPAW parameters, and
warns if the functional, PAW datasets, basis set or mode (e.g. plane-wave
cutoff) differ between competing phases (while k-point and real-space meshes
can differ). Ensure energies use consistent functionals, PAW datasets, basis
sets and cutoffs with the bulk and defects. Competing-phase input generation
with GPAW is not yet implemented (TODO).

The complete MgO example prepares a primitive-cell MgO host calculation, Mg
structures obtained from the |CompetingPhases| workflow and an isolated triplet
O2 reference, then generates Mg-rich and O-rich formation-energy diagrams.

Anisotropic and two-dimensional systems
---------------------------------------

An anisotropic dielectric tensor should be supplied for anisotropic systems.
For example, the graphene regression test uses
``numpy.diag([1e6, 1e6, 1.0])`` to represent metallic in-plane screening and a
vacuum-like out-of-plane response.

Examples and API
----------------

- ``examples/MgO_with_GPAW``: three-dimensional MgO workflow, competing-phase
  references, chemical potentials and formation-energy diagrams.
- ``examples/Graphene_with_GPAW``: anisotropic two-dimensional parsing example.
- :doc:`doped.io`: complete GPAW API reference (``doped.io.gpaw``).

Band edges
----------

Neutral GPAW calculation outputs determine VBM and CBM from the raw band
occupations, independently of the stored Fermi level. Occupations within 0.1
of zero or one are treated as empty or filled, respectively. Partially
occupied states or overlapping occupied and empty levels give a zero gap,
with both edges at the Fermi level. Strong smearing can therefore obscure an
insulating gap; use suitably converged occupation settings for bulk references.
Both occupied and unoccupied bands must be present to determine an insulating
gap. Charged supercells do not supply host band edges; use a neutral bulk
reference. PAW projection mapping and shallow-defect analysis remain separate
work.
