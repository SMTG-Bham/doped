import contextlib
import copy
from pathlib import Path, PurePath
import warnings
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.io.vasp.sets import DictSet
from pymatgen.io.vasp.inputs import Kpoints, UnknownPotcarWarning
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.core import Structure, Composition, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import json 
import pandas as pd

# globally ignore this shit 
warnings.filterwarnings("ignore", category=UnknownPotcarWarning)  
warnings.filterwarnings("ignore", message="No POTCAR file with matching TITEL fields") 
class CompetingPhases(): 
    def __init__(self, system, e_above_hull=0.02): 
        """
        system (list): chemical system under investigation, e.g. ['Mg', 'O']
        e_above_hull (float): maximum considered energy above hull
        """
        # create list of entries 
        molecules_in_a_box = ['H2', 'O2', 'N2', 'F2', 'Cl2', 'Br2']
        # all data collected from materials project
        self.data = ['pretty_formula', 'e_above_hull', 'band_gap', 'nsites', 'volume','icsd_id', "formation_energy_per_atom", 'energy_per_atom', 'energy', 'total_magnetization', 'nelements', 'elements']
        self.system = system
        stype = 'initial'
        m = MPRester()
        self.entries = m.get_entries_in_chemsys(self.system, inc_structure=stype, property_data=self.data)
        self.entries = [e for e in self.entries if e.data['e_above_hull'] <= e_above_hull]
       
        competing_phases = [] 
        # check that none of the elemental ones aren't on the naughty list 
        for e in self.entries: 
            sym = SpacegroupAnalyzer(e.structure) 
            struc = sym.get_primitive_standard_structure()
            if e.data['pretty_formula'] in molecules_in_a_box:
                struc, formula, magnetisation = make_molecule_in_a_box(e.data['pretty_formula']) 
                competing_phases.append({'structure': struc, 'formula': formula, 'formation_energy': 0, 'nsites': 2, 'ehull': 0, 'magnetisation':  magnetisation, 'molecule': True})
            
            else: 
                competing_phases.append({'structure': struc, 'formula': e.data['pretty_formula'], 'formation_energy': e.data["formation_energy_per_atom"], 'nsites': e.data['nsites'], 'ehull': e.data['e_above_hull'], 'magnetisation': e.data['total_magnetization'], 'molecule': False, 'band_gap': e.data['band_gap']})

        # remove make sure it's only unique competing phases 
        self.competing_phases = []
        [self.competing_phases.append(x) for x in competing_phases if x not in self.competing_phases]

    def convergence_setup(self, kpoints_metals=(40,120,5), kpoints_nonmetals=(5,60,5), potcar_functional='PBE_54', user_potcar_settings=None, user_incar_settings=None): 
        # by default uses pbesol, but easy to switch to pbe or 
        # pbe+u by using user_incar_settings
        # user incar settings applies the same settings so both
        file = str(Path(__file__).parent.joinpath('PBEsol_config.json'))
        with open(file) as f: 
            cd = json.load(f)
        
        # kpoints should be set as (min, max, step) 
        min_nm, max_nm, step_nm = kpoints_nonmetals 
        min_m, max_m, step_m = kpoints_metals 

        # separate metals and non-metals
        self.nonmetals = []
        self.metals = []
        for e in self.competing_phases: 
            if not e['molecule']: 
                if e['band_gap'] > 0: 
                    self.nonmetals.append(e)
                else: 
                    self.metals.append(e)
            else: 
                print(f"{e['formula']} is a molecule in a box, does not need convergence testing" )

        for e in self.nonmetals: 
            if user_incar_settings is not None:
                uis = copy.deepcopy(user_incar_settings)
            else: 
                uis = {}
            if e['magnetisation'] > 1: # account for magnetic moment
                if 'ISPIN' not in uis:
                    uis['ISPIN'] = 2
                
            for kpoint in range(min_nm, max_nm, step_nm):
                dis = DictSet(e['structure'], cd,user_potcar_functional=potcar_functional, 
                user_potcar_settings=user_potcar_settings,user_kpoints_settings={'reciprocal_density': kpoint}, user_incar_settings=uis)
                
                kname = 'k'+','.join(str(k) for k in dis.kpoints.kpts[0]) 
                fname = "competing_phases/{}_EaH_{}/kpoint_converge/{}".format(e['formula'], float(f"{e['ehull']:.4f}"), kname)
                dis.write_input(fname)

        for e in self.metals: 
            if user_incar_settings is not None:
                uis = copy.deepcopy(user_incar_settings)
            else: 
                uis = {}
            # change the ismear and sigma for metals 
            uis['ISMEAR'] = -5 
            uis['SIGMA'] = 0.2

            if e['magnetisation'] > 1: # account for magnetic moment
                if 'ISPIN' not in uis:
                    uis['ISPIN'] = 2
                
            for kpoint in range(min_m, max_m, step_m):
                dis = DictSet(e['structure'], cd,user_potcar_functional=potcar_functional, 
                user_potcar_settings=user_potcar_settings,   user_kpoints_settings={'reciprocal_density': kpoint}, user_incar_settings=uis)
                
                kname = 'k'+','.join(str(k) for k in dis.kpoints.kpts[0]) 
                fname = "competing_phases/{}_EaH_{}/kpoint_converge/{}".format(e['formula'], float(f"{e['ehull']:.4f}"), kname)
                dis.write_input(fname)
    
    def vasp_std_setup(self, kpoints_metals=95, kpoints_nonmetals=45, potcar_functional='PBE_54', user_potcar_settings=None, user_incar_settings=None):
        """
        default kpoints 45 for nonmetals, 95 for metals 
        """
        file = str(Path(__file__).parent.joinpath('HSE06_config_relax.json'))
        with open(file) as f: 
            cd = json.load(f)
        
        # separate metals, non-metals and molecules 
        self.nonmetals = []
        self.metals = []
        self.molecules = []
        for e in self.competing_phases:
            if e['molecule']: 
                self.molecules.append(e)
            else:
                if e['band_gap'] > 0: 
                    self.nonmetals.append(e)
                else: 
                    self.metals.append(e)
                
        for e in self.nonmetals: 
            if user_incar_settings is not None:
                uis = copy.deepcopy(user_incar_settings)
            else: 
                uis = {}
            if e['magnetisation'] > 1: # account for magnetic moment
                if 'ISPIN' not in uis:
                    uis['ISPIN'] = 2
                
            dis = DictSet(e['structure'], cd,user_potcar_functional=potcar_functional, user_kpoints_settings={'reciprocal_density': kpoints_nonmetals}, user_incar_settings=uis)
                
            fname = "competing_phases/{}_EaH_{}/vasp_std".format(e['formula'], float(f"{e['ehull']:.4f}"))
            dis.write_input(fname)

        for e in self.metals: 
            if user_incar_settings is not None:
                uis = copy.deepcopy(user_incar_settings)
            else: 
                uis = {}
            # change the ismear and sigma for metals 
            uis['ISMEAR'] = 1 
            uis['SIGMA'] = 0.2

            if e['magnetisation'] > 1: # account for magnetic moment
                if 'ISPIN' not in uis:
                    uis['ISPIN'] = 2
                
            dis = DictSet(e['structure'], cd,user_potcar_functional=potcar_functional, user_kpoints_settings={'reciprocal_density': kpoints_metals}, user_incar_settings=uis)     
            fname = "competing_phases/{}_EaH_{}/vasp_std".format(e['formula'], float(f"{e['ehull']:.4f}"))
            dis.write_input(fname)

        for e in self.molecules:
             
            if user_incar_settings is not None:
                uis = copy.deepcopy(user_incar_settings)
            else: 
                uis = {}
            
            uis['ISIF'] = 2 # can't change the volume

            if e['magnetisation'] > 1: # account for magnetic moment
                if 'ISPIN' not in uis:
                    uis['ISPIN'] = 2
                # idk how many others need nupdown - could use magnetisation key from 
                # the molecule set up to set this automatically? 
                if e['formula'] == 'O2': 
                    uis['NUPDOWN'] = 2
            
            # set up for 2x2x2 kpoints automatically     
            dis = DictSet(e['structure'], cd, user_potcar_functional=potcar_functional, user_kpoints_settings=Kpoints(kpts=[[2,2,2]]), user_incar_settings=uis)
            fname = "competing_phases/{}_EaH_{}/vasp_std".format(e['formula'], float(f"{e['ehull']:.4f}"))
            dis.write_input(fname) 

class AdditionalCompetingPhases(CompetingPhases): 
    """
    if you want to add some extrinsic doping, or add another element to your chemical system, this is the class for you 
    """
    def __init__(self, system, extrinsic_species, e_above_hull=0.02): 
        # the competing phases & entries of the OG system
        super().__init__(system, e_above_hull)
        self.og_competing_phases = copy.deepcopy(self.competing_phases)
        # the competing phases & entries of the OG system + all the additional
        # stuff from the extrinsic species 
        system.append(extrinsic_species)
        super().__init__(system, e_above_hull)
        self.ext_competing_phases = copy.deepcopy(self.competing_phases)

        # only keep the ones that are actually new
        self.competing_phases = []
        for ext in self.ext_competing_phases: 
            if ext not in self.og_competing_phases: 
                self.competing_phases.append(ext)

# separate class for read from file with this as base class? can still use different init?
class CompetingPhasesAnalyzer(): 
    def __init__(self, system): 
        """
            system (string): corresponds to the 'pretty formula' of the bulk composition 
        """
        
        self.bulk_composition = Composition(system)
        self.elemental = [str(c) for c in self.bulk_composition.elements]

    def from_vaspruns(self, path, folder='vasp_std', csv_fname='competing_phases_energies.csv'):   
        """
        read in vaspruns, collate energies to csv; isn't the best at removing higher energy elemental phases (if multiple are present, so double check that)
        Args:
            path (list, str, pathlib Path): either a list of strings or Paths to vasprun.xml(.gz) files, or a path to the base folder in which you have your formula_EaH_/vasp_std/vasprun.xml
            folder (str): the folder in which vasprun is, only use if you set base path (ie. change to vasp_ncl, relax whatever youve called it)
        """
        self.vasprun_paths = [] 
        # fetch data
        # if path is just a list of all competing phases 
        if isinstance(path, list): 
            for p in path: 
                if Path(p).name in {'vasprun.xml', 'vasprun.xml.gz'}:
                    self.vasprun_paths.append(str(Path(p))) 

                # try to find the file - will always pick the first match for vasprun.xml*
                elif len(list(Path(p).glob('vasprun.xml*'))) > 0: 
                    vsp = list(Path(p).glob('vasprun.xml*'))[0]
                    self.vasprun_paths.append(str(vsp))
                
                else:
                    print(f"Can't find a vasprun.xml(.gz) file for {p}, proceed with caution")

        # if path provided points to the doped created directories  
        elif isinstance(path, PurePath) or isinstance(path, str): 
            path = Path(path)
            for p in path.iterdir(): 
                if p.glob('EaH'): 
                    vp = p / folder / 'vasprun.xml'
                    if vp.is_file(): 
                        self.vasprun_paths.append(str(vp))
                    vpg = p / folder / 'vasprun.xml.gz'
                    if vpg.is_file(): 
                        self.vasprun_paths.append(str(vpg))
                    else: 
                        print(f"Can't find a vasprun.xml(.gz) file for {p}, proceed with caution")

                else: 
                    raise FileNotFoundError('Folders are not in the correct structure, provide them as a list of paths (or strings)')

        else: 
            raise ValueError('path should either be a list of paths, a string or a pathlib Path object')
        
        # Ignore POTCAR warnings when loading vasprun.xml
        # pymatgen assumes the default PBE with no way of changing this   
        warnings.filterwarnings("ignore", category=UnknownPotcarWarning)  
        warnings.filterwarnings("ignore", message="No POTCAR file with matching TITEL fields")    
         
        num = len(self.vasprun_paths)
        print(f'parsing {num} vaspruns, this may take a while')
        self.vaspruns = [Vasprun(e).as_dict() for e in self.vasprun_paths]
        self.elemental_vaspruns = []
        self.data = []

        # make a fake dictionary with all elemental energies set to 0 
        temp_elemental_energies = {}
        for e in self.elemental: 
            temp_elemental_energies[e] = 0
        
        # check if elemental, collect the elemental energies per atom (for 
        # formation energies)
        vaspruns_for_removal = []
        for i, v in enumerate(self.vaspruns): 
            comp = [str(c) for c in Composition(v['unit_cell_formula']).elements]
            energy = v['output']['final_energy_per_atom']
            if len(comp) == 1: 
                for key, val in temp_elemental_energies.items(): 
                    if comp[0] == key and energy < val: 
                        temp_elemental_energies[key] = energy
                    elif comp[0] == key and energy > val: 
                        vaspruns_for_removal.append(i)
    
        # get rid of elemental competing phases that aren't the lowest 
        # energy ones 
        if vaspruns_for_removal: 
            for m in sorted(vaspruns_for_removal, reverse=True): 
                del self.vaspruns[m]
        
        temp_data = []
        self.elemental_energies = {}
        for v in self.vaspruns: 
            rcf = v['reduced_cell_formula']
            formulas_per_unit = list(v['unit_cell_formula'].values())[0] / list(rcf.values())[0]
            final_energy = v['output']['final_energy']  

            # check if elemental:
            if len(rcf) == 1: 
                elt = v['elements'][0]
                self.elemental_energies[elt] = v['output']['final_energy_per_atom']

            d = {
                'formula': v['pretty_formula'], 
                'energy_per_fu': final_energy / formulas_per_unit, 
                'energy': final_energy
            }
            temp_data.append(d)
        
        df = _calculate_formation_energies(temp_data, self.elemental_energies)
        df.to_csv(csv_fname, index=False)
        self.data = df.to_dict(orient='records')

    def from_csv(self, csv): 
        """
        read in data from csv. Must have columns 'formula', 'energy_per_fu', 'energy' and 'formation_energy' 
        """
        df = pd.read_csv(csv)
        columns = ['formula', 'energy_per_fu', 'energy', 'formation_energy']
        if all(x in list(df.columns) for x in columns): 

            d = df.to_dict(orient='records')
            self.data = d
        
        else: 
            raise ValueError('supplied csv does not contain the correct headers, cannot read in the data')

    def calculate_chempots(self, include_extrinsic=False, extrinsic_elements=None, csv_fname='chempot_limits.csv'): 
        """
        
        include_extrinsic (bool): whether to mix in extrinsic species in the phase diagram (need to specify elemental in that case, currently doesnt work)
        extrinsic_elements (list): symbols of the extrinsic elements
        csv_fname (str): name of csv file to which chempot limits are saved
        """ 

        self.pd_entries = []
        for d in self.data: 
            e = PDEntry(d['formula'], d['energy_per_fu'])
            self.pd_entries.append(e)

        self.phase_diagram = PhaseDiagram(self.pd_entries, map(Element, self.elemental))
        chem_lims = self.phase_diagram.get_all_chempots(self.bulk_composition)
        chem_limits = {
            'facets': chem_lims, 
            'elemental_refs': {
                elt: ent.energy_per_atom for elt, ent in self.phase_diagram.el_refs.items()
            }, 
            'facets_wrt_el_refs': {}
        }
        # do the shenanigan to relate the facets to the elemental energies
        for facet, chempot_dict in chem_limits['facets'].items(): 
            relative_chempot_dict = copy.deepcopy(chempot_dict)
            for e in relative_chempot_dict.keys(): 
                relative_chempot_dict[e] -= chem_limits['elemental_refs'][e]
            chem_limits['facets_wrt_el_refs'].update({facet: relative_chempot_dict})
        
        # get chemical potentials as pandas dataframe
        self.chemical_potentials = [] 
        for k, v in chem_limits['facets_wrt_el_refs'].items(): 
            lst = []
            columns = []
            for k, v in v.items(): 
                lst.append(v)
                columns.append(str(k))
            self.chemical_potentials.append(lst) 

        df = pd.DataFrame(self.chemical_potentials, columns=columns)
        df.to_csv(csv_fname, index=False)
        print('calculated chemical potential limits: \n')
        print(df)
        
        if include_extrinsic: 
            # 1. work out the formation energies of all dopant competing
            #    phases using the elemental energies
            # 2. for each of the chempots already calculated work out what 
            #    the chemical potential of the dopant would be from
            #       mu_dopant = Hf(dopant competing phase) - sum(mu_elements) 
            # 3. find the most negative mu_dopant which then becomes the new
            #    canonical chemical potential for that dopant species and the 
            #    competing phase is the 'limiting phase' right? 
            # 4. update the chemical potential limits table to reflect this?
            pass


    def cplap_input(self, dependent_variable=None): 
        """ for completeness' sake, automatically saves to  input.dat
        dependent variable (str) gotta pick one of the variables as dependent, the first element is chosen from the compositoin if this isnt set
        """
        with open('input.dat', 'w') as f:
            with contextlib.redirect_stdout(f):
                for i in self.data: 
                    comp = Composition(i['formula']).as_dict()
                    if self.bulk_composition.as_dict() == comp: 
                        print(len(comp))
                        for k, v in comp.items(): 
                            print(int(v), k, end=' ') 
                        print(i['formation_energy'])
                if dependent_variable is not None: 
                    print(dependent_variable)
                else: 
                    print(self.elemental[0])
                print(len(self.data) - 1 - len(self.elemental))
                for i in self.data: 
                    comp = Composition(i['formula']).as_dict()
                    if self.bulk_composition.as_dict() != comp and len(comp) != 1:
                        print(len(comp))
                        for k, v in comp.items(): 
                            print(int(v), k, end=' ') 
                        print(i['formation_energy']) 

def make_molecule_in_a_box(element): 
    # KB: the magnetisation for all other elemental phases is probably wrong - havent checked it myself if i'm completely honest; i guess this will get added to 
    # - keep it above 1 and it will work fine 
    # (but do try to fix it so that the nupdown is the same as magnetisation
    # so that it makes that assignment easier later on when making files)
    # the bond distances are taken from various sources and *not* thoroughly vetted
    lattice = [[30,0,0],[0,30,0],[0,0,30]]
    all_structures = {
        'O2': {'structure': Structure(lattice=lattice, species=['O', 'O'], 
        coords=[[15,15,15], [15,15,16.22]], coords_are_cartesian=True), 'formula': 'O2', 'magnetisation': 2}, 
        'N2': {'structure': Structure(lattice=lattice, species=['N', 'N'], 
        coords=[[15,15,15], [15,15,16.09]], coords_are_cartesian=True), 'formula': 'N2', 'magnetisation': 2}, 
        'H2': {'structure': Structure(lattice=lattice, species=['H', 'H'], 
        coords=[[15,15,15], [15,15,15.74]], coords_are_cartesian=True), 'formula': 'H2', 'magnetisation': 2},
        'F2': {'structure': Structure(lattice=lattice, species=['F', 'F'], 
        coords=[[15,15,15], [15,15,16.44]], coords_are_cartesian=True), 'formula': 'F2', 'magnetisation': 2},
        'Cl2': {'structure': Structure(lattice=lattice, species=['Cl', 'Cl'], coords=[[15,15,15], [15,15,16.99]], coords_are_cartesian=True), 'formula': 'Cl2', 'magnetisation': 2},
        'Br2': {'structure': Structure(lattice=lattice, species=['Br', 'Br'],coords=[[15,15,15], [15,15,17.30]], coords_are_cartesian=True), 'formula': 'Br2', 'magnetisation': 2},
        'I2': {'structure': Structure(lattice=lattice, species=['I', 'I'], 
        coords=[[15,15,15], [15,15,17.67]], coords_are_cartesian=True), 'formula': 'I2', 'magnetisation': 2}
    }
    if element in all_structures.keys(): 
        structure = all_structures[element]['structure']
        formula = all_structures[element]['formula']
        magnetisation = all_structures[element]['magnetisation']

    return structure, formula, magnetisation 


def _calculate_formation_energies(data, elemental): 
    df = pd.DataFrame(data)
    for d in data: 
        for e in elemental.keys(): 
            d[e] = Composition(d['formula']).as_dict()[e] 
        
    df2 = pd.DataFrame(data)
    df2['formation_energy'] = df2['energy_per_fu']
    for k, v in elemental.items(): 
        df2['formation_energy'] -= df2[k] * v 
    
    df['formation_energy'] = df2['formation_energy']
    return df 