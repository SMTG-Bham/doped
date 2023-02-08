import copy
import os
from pathlib import Path
import numpy as np
import unittest
import warnings
from unittest.mock import patch
from doped.competing_phases import CompetingPhases, CompetingPhasesAnalyzer, AdditionalCompetingPhases, make_molecule_in_a_box, _calculate_formation_energies, combine_extrinsic
from pymatgen.core.structure import Structure
from monty.serialization import loadfn, dumpfn

class ChemPotsTestCase(unittest.TestCase): 
    def setUp(self): 
        self.path = Path(__file__).parents[1].joinpath('examples/competing_phases')
        self.stable_system = 'ZrO2'
        self.unstable_system = 'Zr2O'
        self.extrinsic_species = 'La'
        self.csv_path = self.path / 'zro2_competing_phase_energies.csv'
        self.csv_path_ext = self.path / 'zro2_la_competing_phase_energies.csv'

    def tearDown(self) -> None:
        if Path('chempot_limits.csv').is_file(): 
            os.remove('chempot_limits.csv')

        if Path('competing_phases.csv').is_file(): 
            os.remove('competing_phases.csv')
        
        if Path('input.dat').is_file(): 
            os.remove('input.dat')
        
        return super().tearDown()
    
    def test_cpa_csv(self): 
        stable_cpa = CompetingPhasesAnalyzer(self.stable_system)
        stable_cpa.from_csv(self.csv_path)
        self.ext_cpa = CompetingPhasesAnalyzer(self.stable_system, self.extrinsic_species)
        self.ext_cpa.from_csv(self.csv_path_ext)

        self.assertEqual(len(stable_cpa.elemental), 2)
        self.assertEqual(len(self.ext_cpa.elemental), 3)
        self.assertEqual(stable_cpa.data[0]['formula'], 'O2')
        self.assertEqual(self.ext_cpa.data[-1]['formula'], 'La2Zr2O7')
        self.assertAlmostEqual(self.ext_cpa.data[-1]['energy_per_fu'], -119.619571095)

    # test chempots 
    def test_cpa_chempots(self): 
        stable_cpa = CompetingPhasesAnalyzer(self.stable_system)
        stable_cpa.from_csv(self.csv_path)
        df1 = stable_cpa.calculate_chempots()
        self.assertEqual(list(df1['O'])[0], 0)
        #check if it's no longer Element
        self.assertEqual(type(list(stable_cpa.intrinsic_chem_limits['elemental_refs'].keys())[0]), str)

        self.unstable_cpa = CompetingPhasesAnalyzer(self.unstable_system)
        self.unstable_cpa.from_csv(self.csv_path)
        with self.assertRaises(ValueError): 
            self.unstable_cpa.calculate_chempots()

        self.ext_cpa = CompetingPhasesAnalyzer(self.stable_system, self.extrinsic_species)
        self.ext_cpa.from_csv(self.csv_path_ext)
        df = self.ext_cpa.calculate_chempots()
        self.assertEqual(list(df['La_limiting_phase'])[0], 'La2Zr2O7')
        self.assertAlmostEqual(list(df['La'])[0], -9.46298748)

    # test vaspruns 
    def test_vaspruns(self): 
        cpa = CompetingPhasesAnalyzer(self.stable_system) 
        path = self.path / 'ZrO'
        cpa.from_vaspruns(path=path, folder='relax', csv_fname=self.csv_path) 
        self.assertEqual(len(cpa.elemental), 2) 
        self.assertEqual(cpa.data[0]['formula'], 'O2')

        cpa_no = CompetingPhasesAnalyzer(self.stable_system)
        with self.assertRaises(FileNotFoundError):
            cpa_no.from_vaspruns(path='path')
        
        with self.assertRaises(ValueError): 
            cpa_no.from_vaspruns(path=0)
        
        ext_cpa = CompetingPhasesAnalyzer(self.stable_system, self.extrinsic_species)
        path2 = self.path / 'LaZrO'
        ext_cpa.from_vaspruns(path=path2, folder='relax', csv_fname=self.csv_path_ext)
        self.assertEqual(len(ext_cpa.elemental), 3)
        self.assertEqual(ext_cpa.data[1]['formula'], 'Zr')

        # check if it works from a list
        all_paths = []
        for p in path.iterdir():
            if not p.name.startswith('.'): 
                pp = p / 'relax' / 'vasprun.xml' 
                ppgz = p / 'relax' / 'vasprun.xml.gz' 
                if pp.is_file():
                    all_paths.append(pp)
                elif ppgz.is_file(): 
                    all_paths.append(ppgz) 
        lst_cpa = CompetingPhasesAnalyzer(self.stable_system) 
        lst_cpa.from_vaspruns(path=all_paths)
        self.assertEqual(len(lst_cpa.elemental), 2)   
        self.assertEqual(len(lst_cpa.vasprun_paths), 8)   

        all_fols = []
        for p in path.iterdir():
            if not p.name.startswith('.'): 
                pp = p / 'relax' 
                all_fols.append(pp)
        lst_fols_cpa = CompetingPhasesAnalyzer(self.stable_system)
        lst_fols_cpa.from_vaspruns(path=all_fols)
        self.assertEqual(len(lst_fols_cpa.elemental), 2)

    def test_cplap_input(self): 
        cpa = CompetingPhasesAnalyzer(self.stable_system)
        cpa.from_csv(self.csv_path)
        cpa.cplap_input(dependent_variable='O')

        self.assertTrue(Path('input.dat').is_file())

        with open('input.dat', 'r') as f: 
            contents = f.readlines()
            self.assertEqual(contents[5], '3 Zr 1 O -5.986573519999993\n')


class BoxedMoleculesTestCase(unittest.TestCase): 
    def test_elements(self): 
        s,f,m = make_molecule_in_a_box('O2')
        self.assertEqual(f, 'O2')
        self.assertEqual(m, 2)
        self.assertEqual(type(s), Structure)
        
        with self.assertRaises(UnboundLocalError): 
            s2, f2, m2 = make_molecule_in_a_box('R2')


class FormationEnergyTestCase(unittest.TestCase):
    def test_fe(self):  
        elemental = {'O': -7.006602065, 'Zr': -9.84367624}
        data = [{'formula': 'O2',
                'energy_per_fu': -14.01320413,
                'energy_per_atom': -7.006602065,
                'energy': -14.01320413},
                {'formula': 'Zr',
                'energy_per_fu': -9.84367624,
                'energy_per_atom': -9.84367624,
                'energy': -19.68735248},
                {'formula': 'Zr3O',
                'energy_per_fu': -42.524204305,
                'energy_per_atom': -10.63105107625,
                'energy': -85.04840861},
                {'formula': 'ZrO2',
                'energy_per_fu': -34.5391058,
                'energy_per_atom': -11.5130352,
                'energy': -138.156423},
                {'formula': 'ZrO2',
                'energy_per_fu': -34.83230881,
                'energy_per_atom': -11.610769603333331,
                'energy': -139.32923524},
                {'formula': 'Zr2O',
                'energy_per_fu': -32.42291351666667,
                'energy_per_atom': -10.807637838888889,
                'energy': -194.5374811}]
        
        df = _calculate_formation_energies(data, elemental)
        self.assertNotEqual(len(df['formula']), len(data)) #check that it removes 1
        self.assertAlmostEqual(df['formation_energy'][4], -5.728958971666668 )
        self.assertEqual(df[df['formula']=='O2']['formation_energy'][0], 0)

class CombineExtrinsicTestCase(unittest.TestCase): 
    def setUp(self) -> None:
        self.path = Path(__file__).parents[1].joinpath('examples/competing_phases')
        first = self.path / 'zro2_la_chempots.json'
        second = self.path / 'zro2_y_chempots.json'
        self.first = loadfn(first)
        self.second = loadfn(second)
        self.extrinsic_species = 'Y'
        return super().setUp()
    
    def test_combine_extrinsic(self): 
        d = combine_extrinsic(self.first, self.second, self.extrinsic_species) 
        self.assertEqual(len(d['elemental_refs'].keys()), 4)
        facets = list(d['facets'].keys())
        self.assertEqual(facets[0].rsplit('-', 1)[1], 'Y2Zr2O7')
    
    def test_combine_extrinsic_errors(self): 
        d = {'a': 1}
        with self.assertRaises(KeyError): 
            combine_extrinsic(d, self.second, self.extrinsic_species)
        
        with self.assertRaises(KeyError): 
            combine_extrinsic(self.first, d, self.extrinsic_species)
        
        with self.assertRaises(ValueError): 
            combine_extrinsic(self.first, self.second, 'R')
