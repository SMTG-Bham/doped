"""Generate GPAW inputs for intrinsic and nitrogen-related graphene defects."""

from pathlib import Path

from pymatgen.core import Structure

from doped.generation import DefectsGenerator
from doped.io.gpaw import DefectsSet


def main():
    script_dir = Path(__file__).resolve().parent
    graphene_structure = Structure.from_file(script_dir / "graphene4x4.cif")

    print("Generating graphene defects...")
    defect_generator = DefectsGenerator(
        graphene_structure,
        extrinsic="N",
        generate_supercell=False,
        charge_state_gen_kwargs={"probability_threshold": 0.01},
    )

    gpaw_settings = {
        "mode": {"name": "pw", "ecut": 200},
        "xc": "PBE",
        "kpts": {"size": (1, 1, 1), "gamma": True},
        "fmax": 0.05,
    }

    # Write a ``<defect name>/relax.py`` relaxation for each defect, and a static reference calculation for
    # the bulk supercell (``C_bulk/singlepoint.py``):
    print("Writing GPAW inputs...")
    DefectsSet(defect_generator, gpaw_settings=gpaw_settings).write_files()

    print("Graphene GPAW workflow setup complete.")


if __name__ == "__main__":
    main()
