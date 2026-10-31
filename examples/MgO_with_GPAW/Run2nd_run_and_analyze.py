"""Parse the GPAW results and plot MgO defect formation energies."""

from pathlib import Path

from monty.serialization import dumpfn

from doped.chemical_potentials import CompetingPhasesAnalyzer
from doped.parsing import DefectsParser
from doped.thermodynamics import DefectThermodynamics

DIELECTRIC = 8.8963


def main():
    """Parse GPAW calculations and write Mg-rich and O-rich formation-energy plots."""
    calculation_dir = Path.cwd()

    print("Parsing MgO defect calculations...")
    defect_parser = DefectsParser(
        output_path=calculation_dir,
        bulk_path="MgO_bulk",
        dielectric=DIELECTRIC,
        calculator="gpaw",
    )

    print("Building GPAW-consistent Mg/O chemical potentials...")
    phase_analyzer = CompetingPhasesAnalyzer(
        "MgO", entries=calculation_dir / "CompetingPhases_GPAW", calculator="gpaw"
    )
    chempots = phase_analyzer.chempots
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
