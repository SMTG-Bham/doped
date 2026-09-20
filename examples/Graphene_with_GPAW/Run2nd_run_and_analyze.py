"""Parse completed graphene GPAW calculations with anisotropic screening."""

from pathlib import Path

import numpy as np
from monty.serialization import dumpfn

from doped.gpaw import GPAWDefectsParser

DIELECTRIC = np.diag([1e6, 1e6, 1.0])


def main():
    calculation_dir = Path.cwd()

    print("Parsing graphene defect calculations...")
    parser = GPAWDefectsParser(
        output_path=calculation_dir,
        bulk_path="bulk",
        dielectric=DIELECTRIC,
    )
    defect_dict = parser.defect_dict
    dumpfn(defect_dict, calculation_dir / "Graphene_GPAW_defect_dict.json.gz")

    print(f"Parsed {len(defect_dict)} defects.")
    for defect_name, defect_entry in sorted(defect_dict.items()):
        correction = defect_entry.corrections.get("kumagai_charge_correction")
        correction_text = "None" if correction is None else f"{float(correction):.6f} eV"
        print(
            f"{defect_name}: charge={defect_entry.charge_state:+d}, "
            f"eFNV correction={correction_text}"
        )

    print(
        "The anisotropic eFNV results demonstrate parser handling for a 2D "
        "cell; use a correction derived for 2D boundary conditions for "
        "quantitative formation energies."
    )


if __name__ == "__main__":
    main()
