
import unittest
import os
import shutil
from pymatgen.core.structure import Structure
from doped.gpaw import GPAWDefectRelaxSet
from doped.generation import DefectsGenerator

class GPAWTest(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.output_dir = "gpaw_test_outputs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Create a simple structure for testing
        self.structure = Structure.from_file(os.path.join(self.data_dir, "Cu_prim_POSCAR"))

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_gpaw_defect_relax_set(self):
        # Test with Structure
        relax_set = GPAWDefectRelaxSet(self.structure, charge_state=1)
        relax_set.write_input(self.output_dir)
        
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "relax.py")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "structure.cif")))
        
        with open(os.path.join(self.output_dir, "relax.py"), "r") as f:
            content = f.read()
            self.assertIn("charge=1", content)
            self.assertIn("mode=PW(ecut=400)", content) # Default

    def test_gpaw_defect_relax_set_custom(self):
        # Test with custom settings
        gpaw_settings = {
            "mode": {"name": "pw", "ecut": 300},
            "xc": "PBE",
            "kpts": {"size": (2, 2, 2), "gamma": True},
        }
        relax_set = GPAWDefectRelaxSet(self.structure, charge_state=-1, gpaw_settings=gpaw_settings)
        relax_set.write_input(self.output_dir)
        
        with open(os.path.join(self.output_dir, "relax.py"), "r") as f:
            content = f.read()
            self.assertIn("charge=-1", content)
            self.assertIn("mode=PW(ecut=300)", content)
            self.assertIn("'size': (2, 2, 2)", content)

if __name__ == "__main__":
    unittest.main()
