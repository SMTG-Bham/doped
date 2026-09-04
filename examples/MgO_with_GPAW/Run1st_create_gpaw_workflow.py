from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from doped.generation import DefectsGenerator
from doped.gpaw import GPAWDefectRelaxSet


def main():
    print("Generating MgO bulk structure natively...")
    # Native structure generation
    lattice = Lattice.cubic(4.21)
    bulk_structure = Structure.from_spacegroup(
        "Fm-3m",
        lattice,
        ["Mg", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )

    print("Generating MgO defects (forcing Mg_O antisites)...")
    # Force the generation of the Mg_O antisite using the extrinsic flag
    defect_gen = DefectsGenerator(bulk_structure, extrinsic={"O": "Mg"})

    # GPAW Parameters
    gpaw_settings = {
        "mode": {"name": "pw", "ecut": 250},
        "kpts": {"size": (1, 1, 1), "gamma": True},
        "xc": "PBE",
    }

    print("Writing GPAW input files...")

    # Setup Bulk using the finalized API parameters
    bulk_set = GPAWDefectRelaxSet(
        defect_gen.bulk_supercell,
        charge_state=0,
        gpaw_settings=gpaw_settings,
    )
    bulk_set.write_input("bulk")

    # Setup Defects
    for defect_entry in defect_gen.defect_entries.values():
        defect_name = f"{defect_entry.defect.name}_{defect_entry.charge_state:+d}"
        # Example of filtering defects:
        # if "v_Mg" not in defect_name and "Mg_O" not in defect_name:
        #     continue

        print(f"Setting up {defect_name}...")

        # Pass the entry directly, and use the gpaw_settings dictionary
        defect_set = GPAWDefectRelaxSet(
            defect_entry,
            charge_state=defect_entry.charge_state,
            gpaw_settings=gpaw_settings,
        )
        defect_set.write_input(defect_name)

        # Create a true static single-point example for the Mg_O +1 state
        if "Mg_O" in defect_name and defect_entry.charge_state == 1:
            unrelaxed_name = defect_name + "_unrelaxed"
            print(f"Setting up {unrelaxed_name}...")

            singlepoint_set = GPAWDefectRelaxSet(
                defect_entry,
                charge_state=defect_entry.charge_state,
                gpaw_settings=gpaw_settings,
                calculation_type="singlepoint",
            )
            singlepoint_set.write_input(unrelaxed_name)

    print("Workflow setup complete! You can now run the calculations.")


if __name__ == "__main__":
    main()
