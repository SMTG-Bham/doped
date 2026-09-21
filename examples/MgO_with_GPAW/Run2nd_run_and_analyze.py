"""Parse the GPAW results and plot MgO defect formation energies."""

from pathlib import Path

from monty.serialization import dumpfn

from doped.chemical_potentials import get_doped_chempots_from_entries
from doped.io import get_calculation_outputs
from doped.parsing import DefectsParser
from doped.thermodynamics import DefectThermodynamics

DIELECTRIC = 8.8963


def get_gpaw_entry(path):
    """Read one GPAW calculation as a computed structure entry."""
    return get_calculation_outputs(path, calculator="gpaw").get_computed_entry()


def main():
    calculation_dir = Path.cwd()

    print("Parsing MgO defect calculations...")
    defect_parser = DefectsParser(
        output_path=calculation_dir,
        bulk_path="bulk",
        dielectric=DIELECTRIC,
        calculator="gpaw",
    )

    print("Building GPAW-consistent Mg/O chemical potentials...")
    bulk_entry = get_gpaw_entry(calculation_dir / "bulk")
    phase_dir = calculation_dir / "CompetingPhases_GPAW"
    phase_entries = [get_gpaw_entry(path) for path in sorted(phase_dir.iterdir()) if path.is_dir()]
    chempots = get_doped_chempots_from_entries(
        [bulk_entry, *phase_entries],
        bulk_entry,
    )
    dumpfn(chempots, calculation_dir / "MgO_GPAW_chempots.json")

    defect_thermodynamics = DefectThermodynamics(
        defect_parser.defect_dict,
        chempots=chempots,
    )
    defect_thermodynamics.to_json(calculation_dir / "MgO_GPAW_DefectThermodynamics.json.gz")

    for limit in ("Mg-rich", "O-rich"):
        figure = defect_thermodynamics.plot(limit=limit)
        filename = calculation_dir / f"MgO_formation_energies_{limit}.png"
        figure.savefig(filename, bbox_inches="tight", dpi=300)
        print(f"Wrote {filename}")


if __name__ == "__main__":
    main()
