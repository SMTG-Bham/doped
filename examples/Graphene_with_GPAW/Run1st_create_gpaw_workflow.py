"""Generate GPAW inputs for intrinsic and nitrogen-related graphene defects."""

from pathlib import Path

from pymatgen.core import Structure

from doped.generation import DefectsGenerator
from doped.gpaw import GPAWDefectRelaxSet


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

    print("Writing bulk GPAW input...")
    bulk_set = GPAWDefectRelaxSet(
        defect_generator.bulk_supercell,
        charge_state=0,
        gpaw_settings=gpaw_settings,
    )
    bulk_set.write_input("bulk")

    print("Writing defect GPAW inputs...")
    for defect_name, defect_entry in defect_generator.defect_entries.items():
        print(f"Setting up {defect_name}...")
        defect_set = GPAWDefectRelaxSet(
            defect_entry,
            charge_state=defect_entry.charge_state,
            gpaw_settings=gpaw_settings,
        )
        defect_set.write_input(defect_name)

    print("Graphene GPAW workflow setup complete.")


if __name__ == "__main__":
    main()
