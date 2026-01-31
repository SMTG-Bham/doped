
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
        relax_set = GPAWDefectRelaxSet(self.structure, charge_state=-1, gpaw_settings=gpaw_settings, poscar_comment="Test Comment")
        relax_set.write_input(self.output_dir)
        
        self.assertEqual(relax_set.poscar_comment, "Test Comment")
        
        with open(os.path.join(self.output_dir, "relax.py"), "r") as f:
            content = f.read()
            self.assertIn("charge=-1", content)
            self.assertIn("mode=PW(ecut=300)", content)
            self.assertIn("'size': (2, 2, 2)", content)
            self.assertIn("from gpaw import GPAW, PW, LCAO, FD", content)

    def test_gpaw_defect_relax_set_lcao(self):
        # Test with LCAO mode
        gpaw_settings = {
            "mode": {"name": "lcao", "basis": "dzp"},
        }
        relax_set = GPAWDefectRelaxSet(self.structure, charge_state=0, gpaw_settings=gpaw_settings)
        relax_set.write_input(self.output_dir)
        
        with open(os.path.join(self.output_dir, "relax.py"), "r") as f:
            content = f.read()
            self.assertIn("mode=LCAO(basis='dzp')", content)
            self.assertIn("from gpaw import GPAW, PW, LCAO, FD", content)

    def test_gpaw_parser(self):
        from unittest.mock import MagicMock, patch
        from doped.gpaw import GPAWParser
        
        # Mock GPAW calculator and atoms
        with patch("gpaw.GPAW") as mock_gpaw:
            mock_calc = MagicMock()
            mock_gpaw.return_value = mock_calc
            
            mock_atoms = MagicMock()
            mock_calc.get_atoms.return_value = mock_atoms
            mock_calc.get_potential_energy.return_value = -10.5
            mock_calc.get_fermi_level.return_value = 2.0
            mock_calc.get_number_of_spins.return_value = 1
            mock_calc.get_ibz_k_points.return_value = [0]
            mock_calc.get_eigenvalues.return_value = [1.0, 1.5, 2.5, 3.0]
            
            # Mock AseAtomsAdaptor
            with patch("pymatgen.io.ase.AseAtomsAdaptor.get_structure") as mock_get_struct:
                mock_get_struct.return_value = self.structure
                
                parser = GPAWParser("dummy.gpw")
                
                self.assertEqual(parser.energy, -10.5)
                self.assertEqual(parser.structure, self.structure)
                
                band_gap, cbm, vbm, efermi = parser.get_eigenvalue_properties()
                self.assertEqual(efermi, 2.0)
                self.assertEqual(vbm, 1.5)
                self.assertEqual(cbm, 2.5)
                self.assertEqual(band_gap, 1.0)
                
                # Test get_computed_entry
                entry = parser.get_computed_entry()
                self.assertEqual(entry.energy, -10.5)
                self.assertEqual(entry.composition, self.structure.composition)

if __name__ == "__main__":
    unittest.main()
