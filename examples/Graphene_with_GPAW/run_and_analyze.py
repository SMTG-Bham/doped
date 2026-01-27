
import os
import glob
import subprocess
from doped.gpaw import GPAWDefectsParser
from doped.analysis import DefectThermodynamics
import shutil

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
bulk_dir = os.path.join(base_dir, "calculation_bulk")
CORES = 8 # Set number of cores for MPI

def run_calculation(folder):
    """Runs the relax.py script in the given folder."""
    print(f"Processing {folder}...")
    relax_script = os.path.join(folder, "relax.py")
    
    # Check if already run (relaxed.gpw exists)
    if os.path.exists(os.path.join(folder, "relaxed.gpw")):
        print(f"  relaxed.gpw exists, skipping calculation.")
        return

    # Check if failed/non-conserving (gpaw_output.txt exists but relaxed.gpw doesn't)
    if os.path.exists(os.path.join(folder, "gpaw_output.txt")):
        print(f"  gpaw_output.txt exists but relaxed.gpw does not. Skipping likely non-conserving/failed calculation.")
        return

    if not os.path.exists(relax_script):
        print(f"  relax.py not found in {folder}, skipping.")
        return

    try:
        # Construct command
        if CORES > 1:
            # Use mpirun for parallel execution
            # Note: gpaw python is often recommended but python usually works if gpaw installed correctly
            cmd = ["mpirun", "-np", str(CORES), "gpaw", "python", "relax.py"]
        else:
            cmd = ["python", "relax.py"]

        # Run in the folder to handle relative paths correctly
        subprocess.run(cmd, cwd=folder, check=True, stdout=subprocess.DEVNULL)
        print(f"  Calculation completed.")
    except subprocess.CalledProcessError as e:
        print(f"  Calculation failed: {e}")
    except FileNotFoundError:
        print(f"  Execution failed. Check if mpirun/gpaw are in your PATH.")
    except Exception as e:
        print(f"  An error occurred: {e}")

# 1. Run Bulk Calculation
print("--- Step 1: Running Bulk Calculation ---")
run_calculation(bulk_dir)

# 2. Run Defect Calculations
print("\n--- Step 2: Running Defect Calculations ---")
defect_folders = sorted(glob.glob(os.path.join(base_dir, "calculation_*")))
defect_folders = [f for f in defect_folders if f != bulk_dir]

# Run all defects
for folder in defect_folders:
    run_calculation(folder)

# 3. Parse Results with Doped
print("\n--- Step 3: Parsing Results with Doped ---")
try:
    parser = GPAWDefectsParser(
        output_path=base_dir,
        bulk_path=bulk_dir,
        dielectric=10  # Dummy dielectric for correction testing
    )
    defect_dict = parser.parse_all()
    
    print(f"Parsed {len(defect_dict)} defects.")
    
    if defect_dict:
        # 4. Thermodynamic Analysis (Demo)
        print("\n--- Step 4: Thermodynamic Analysis ---")
        thermo = DefectThermodynamics(list(defect_dict.values()))
        
        # We can't easily plot without X server, but we can print formation energies
        # at a specific Fermi level (e.g., VBM)
        for defect_name, defect_entry in defect_dict.items():
             # Simple formation energy calculation: E_defect - E_bulk + q*E_fermi
             # This ignores chemical potentials for now as we didn't calculate them.
             # Doped's get_formation_energy needs chemical potentials.
             
             # Let's print the raw energy difference and corrections
             e_form_raw = defect_entry.formation_energy(fermi_level=0, chempots={})
             print(f"Defect: {defect_name}")
             print(f"  Charge: {defect_entry.charge_state}")
             print(f"  Supercell Energy: {defect_entry.sc_entry_energy:.4f} eV")
             print(f"  Corrections: {defect_entry.corrections}")
             print(f"  Formation Energy (at VBM, no chempots): {e_form_raw:.4f} eV")
             
except Exception as e:
    print(f"Analysis failed: {e}")
    import traceback
    traceback.print_exc()

print("\nFull Automation Test Complete.")


