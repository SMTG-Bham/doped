
import os
import shutil
from pymatgen.core.structure import Structure
from doped.generation import DefectsGenerator
from doped.gpaw import GPAWDefectRelaxSet

# 1. Load the provided defect structure
script_dir = os.path.dirname(os.path.abspath(__file__))
defect_file = os.path.join(script_dir, "graphene4x4.cif")
try:
    defect_structure = Structure.from_file(defect_file)
    print(f"Loaded structure from {defect_file}")
except Exception as e:
    print(f"Failed to load {defect_file}: {e}")
    exit(1)

# 2. Create GPAW input for the provided defect structure directly
print("Generating GPAW input for the provided defect structure...")

# SETTINGS
gpaw_settings = {
    "mode": {"name": "pw", "ecut": 200}, 
    "xc": "PBE",
    "kpts": {"size": (1, 1, 1), "gamma": True},
    "txt": "gpaw_output.txt",
    "fmax": 5.0 # Very loose relaxation for speed test
}

relax_set = GPAWDefectRelaxSet(defect_structure, charge_state=0, gpaw_settings=gpaw_settings)
relax_set.write_input(os.path.join(script_dir, "calculation_original_defect"))
print("Input written to calculation_original_defect/")

# 3. Attempt to create a 'bulk' structure by replacing N with C
# Assuming the defect is a single N substitution for C in Graphene
print("\nAttempting to create bulk structure for defect generation...")
bulk_structure = defect_structure.copy()
bulk_structure.replace_species({"N": "C"})
print(f"Bulk composition: {bulk_structure.composition}")

# WRITE BULK INPUT
bulk_folder = os.path.join(script_dir, "calculation_bulk")
relax_set = GPAWDefectRelaxSet(bulk_structure, charge_state=0, gpaw_settings=gpaw_settings, poscar_comment="Bulk Structure")
relax_set.write_input(bulk_folder)
print(f"Input written to {bulk_folder}/")

# 4. Generate defects using DefectsGenerator
# We will generate vacancies and substitutions (N)
print("\nGenerating defects from derived bulk structure...")
try:
    generator = DefectsGenerator(
        bulk_structure,
        extrinsic="N",
        generate_supercell=False, # We already have a supercell
        charge_state_gen_kwargs={"probability_threshold": 0.01}
    )
    
    print("Generated defects:")
    for name, entry in generator.defect_entries.items():
        print(f" - {name}")
        
    # 5. Write GPAW inputs for all generated defects
    print("\nWriting GPAW inputs for generated defects...")
    for name, entry in generator.defect_entries.items():
        folder_name = os.path.join(script_dir, f"calculation_{name}")
        # Determine charge state if possible, otherwise default to 0
        charge = entry.charge_state
        
        relax_set = GPAWDefectRelaxSet(entry, charge_state=charge, gpaw_settings=gpaw_settings)
        relax_set.write_input(folder_name)
        print(f" - Written to {folder_name}/")

except Exception as e:
    print(f"Defect generation failed: {e}")
    import traceback
    traceback.print_exc()

print("\nWorkflow complete.")
