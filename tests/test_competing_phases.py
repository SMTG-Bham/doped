import os
from pathlib import Path
import numpy as np
import unittest
import warnings
from unittest.mock import patch
from doped.competing_phases import CompetingPhases, CompetingPhasesAnalyzer, AdditionalCompetingPhases

class ChemPotsTestCase(unittest.TestCase): 
    def setUp(self): 
        self.path = Path(__file__).parents[1].joinpath('examples/competing_phases')
        self.stable_system = 'ZrO2'
        self.unstable_system = 'Zr2O'
        self.extrinsic_species = 'La'

        
    def test_cpa_csv(self): 
        csv_path = self.path / 'zro2_competing_phase_energies.csv'
        csv_path_ext = self.path / 'zro2_la_competing_phase_energies.csv'
        self.stable_cpa = CompetingPhasesAnalyzer(self.stable_system)
        self.stable_cpa.from_csv(csv_path)
        self.ext_cpa = CompetingPhasesAnalyzer(self.stable_system, self.extrinsic_species)
        self.ext_cpa.from_csv(csv_path_ext)

        self.assertEqual(len(self.stable_cpa.elemental), 2)
        self.assertEqual(len(self.ext_cpa.elemental), 3)
        self.assertEqual(self.stable_cpa.data[0]['formula'], 'O2')
        self.assertEqual(self.ext_cpa.data[-1]['formula'], 'La2O3')
        self.assertAlmostEqual(self.ext_cpa.data[-1]['energy_per_fu'], -49.046189555)

    # test chempots 
    def test_cpa_chempots(self): 
        csv_path = self.path / 'zro2_competing_phase_energies.csv'
        csv_path_ext = self.path / 'zro2_la_competing_phase_energies.csv'
        self.stable_cpa = CompetingPhasesAnalyzer(self.stable_system)
        self.stable_cpa.from_csv(csv_path)
        df1 = self.stable_cpa.calculate_chempots()
        self.assertEqual(list(df1['O'])[0], 0)

        self.unstable_cpa = CompetingPhasesAnalyzer(self.unstable_system)
        self.unstable_cpa.from_csv(csv_path)
        with self.assertRaises(ValueError): 
            self.unstable_cpa.calculate_chempots()

        self.ext_cpa = CompetingPhasesAnalyzer(self.stable_system, self.extrinsic_species)
        self.ext_cpa.from_csv(csv_path_ext)
        df = self.ext_cpa.calculate_chempots()
        self.assertEqual(list(df['La_limiting_phase'])[0], 'La2Zr2O7')
        self.assertAlmostEqual(list(df['La'])[0], -9.46298748)

        if os.path.isfile('chempot_limits.csv'): 
            os.remove('chempot_limits.csv')


    # test vaspruns 

