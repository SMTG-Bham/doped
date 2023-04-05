import contextlib
import copy
import os
import warnings
from pathlib import Path, PurePath

import pandas as pd
from monty.serialization import loadfn
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Element, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Kpoints, UnknownPotcarWarning
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.sets import BadInputSetWarning, DictSet

from doped.pycdt.utils.parse_calculations import get_vasprun

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
default_potcar_dict = loadfn(os.path.join(MODULE_DIR, "PotcarSet.yaml"))

# globally ignore:
warnings.filterwarnings("ignore", category=BadInputSetWarning)
warnings.filterwarnings("ignore", category=UnknownPotcarWarning)
warnings.filterwarnings("ignore", message="No POTCAR file with matching TITEL fields")
warnings.filterwarnings(
    "ignore", message="POTCAR data with symbol"
)  # Ignore POTCAR warnings because Pymatgen incorrectly detecting POTCAR types
warnings.filterwarnings(
    "ignore", message="You are using the legacy MPRester"
)  # currently rely on this so shouldn't show warning
warnings.filterwarnings("ignore", message="Ignoring unknown variable type")


# TODO: Check default error when user attempts `CompetingPhases()` with no API key setup; if not
#  sufficiently informative, add try except catch to give more informative error message for this.
# TODO: Need to recheck all functionality from old `_chemical_potentials.py` is now present here.


def make_molecule_in_a_box(element):
    # (but do try to fix it so that the nupdown is the same as magnetisation
    # so that it makes that assignment easier later on when making files)
    # the bond distances are taken from various sources and *not* thoroughly vetted
    lattice = [[30, 0, 0], [0, 30, 0], [0, 0, 30]]
    all_structures = {
        "O2": {
            "structure": Structure(
                lattice=lattice,
                species=["O", "O"],
                coords=[[15, 15, 15], [15, 15, 16.22]],
                coords_are_cartesian=True,
            ),
            "formula": "O2",
            "total_magnetization": 2,
        },
        "N2": {
            "structure": Structure(
                lattice=lattice,
                species=["N", "N"],
                coords=[[15, 15, 15], [15, 15, 16.09]],
                coords_are_cartesian=True,
            ),
            "formula": "N2",
            "total_magnetization": 0,
        },
        "H2": {
            "structure": Structure(
                lattice=lattice,
                species=["H", "H"],
                coords=[[15, 15, 15], [15, 15, 15.74]],
                coords_are_cartesian=True,
            ),
            "formula": "H2",
            "total_magnetization": 0,
        },
        "F2": {
            "structure": Structure(
                lattice=lattice,
                species=["F", "F"],
                coords=[[15, 15, 15], [15, 15, 16.44]],
                coords_are_cartesian=True,
            ),
            "formula": "F2",
            "total_magnetization": 0,
        },
        "Cl2": {
            "structure": Structure(
                lattice=lattice,
                species=["Cl", "Cl"],
                coords=[[15, 15, 15], [15, 15, 16.99]],
                coords_are_cartesian=True,
            ),
            "formula": "Cl2",
            "total_magnetization": 0,
        },
    }

    if element in all_structures:
        structure = all_structures[element]["structure"]
        formula = all_structures[element]["formula"]
        total_magnetization = all_structures[element]["total_magnetization"]

    return structure, formula, total_magnetization


def _make_molecular_entry(computed_entry):
    """
    Generate a new ComputedStructureEntry for a molecule in a box, for the input elemental
    ComputedEntry
    """
    assert len(computed_entry.composition.elements) == 1  # Elemental!
    struc, formula, total_magnetization = make_molecule_in_a_box(
        computed_entry.data["pretty_formula"]
    )
    molecular_entry = ComputedStructureEntry(
        structure=struc,
        energy=computed_entry.energy_per_atom * 2,  # set entry energy to be hull energy
        composition=Composition(formula),
        parameters=None,
    )
    molecular_entry.data["oxide_type"] = "None"
    molecular_entry.data["pretty_formula"] = formula
    molecular_entry.data["e_above_hull"] = 0
    molecular_entry.data["band_gap"] = None
    molecular_entry.data["nsites"] = 2
    molecular_entry.data["volume"] = 0
    molecular_entry.data["icsd_id"] = None
    molecular_entry.data["formation_energy_per_atom"] = 0
    molecular_entry.data["energy_per_atom"] = computed_entry.data["energy_per_atom"]
    molecular_entry.data["energy"] = computed_entry.data["energy_per_atom"] * 2
    molecular_entry.data["total_magnetization"] = total_magnetization
    molecular_entry.data["nelements"] = 1
    molecular_entry.data["elements"] = [formula]
    molecular_entry.data["molecule"] = True

    return molecular_entry


def _calculate_formation_energies(data, elemental):
    df = pd.DataFrame(data)
    for d in data:
        for e in elemental.keys():
            d[e] = Composition(d["formula"]).as_dict()[e]

    df2 = pd.DataFrame(data)
    df2["formation_energy"] = df2["energy_per_fu"]
    for k, v in elemental.items():
        df2["formation_energy"] -= df2[k] * v

    df["formation_energy"] = df2["formation_energy"]

    df["num_atoms_in_fu"] = df["energy_per_fu"] / df["energy_per_atom"]
    df["num_species"] = df["formula"].apply(lambda x: len(Composition(x).as_dict()))

    # sort by num_species, then alphabetically, then by num_atoms_in_fu, then by formation_energy
    df.sort_values(
        by=["num_species", "formula", "num_atoms_in_fu", "formation_energy"],
        inplace=True,
    )
    # drop num_atoms_in_fu and num_species
    df.drop(columns=["num_atoms_in_fu", "num_species"], inplace=True)

    return df


def _renormalise_entry(entry, renormalisation_energy_per_atom):
    """
    Regenerate the input entry with an energy per atom decreased by renormalisation_energy_per_atom
    """
    renormalised_entry_dict = entry.as_dict().copy()
    renormalised_entry_dict[
        "energy"
    ] = entry.energy - renormalisation_energy_per_atom * sum(
        entry.composition.values()
    )  # entry.energy includes MP corrections as desired
    renormalised_entry = PDEntry.from_dict(renormalised_entry_dict)
    return renormalised_entry


def get_chempots_from_phase_diagram(bulk_ce, phase_diagram):
    """
    Get the chemical potential limits for the bulk computed entry in the supplied phase diagram.

    Args:
        bulk_ce: Pymatgen ComputedStructureEntry object for bulk entry / supercell
        phase_diagram: Pymatgen PhaseDiagram object for the system of interest
    """
    bulk_composition = bulk_ce.composition
    redcomp = bulk_composition.reduced_composition
    # append bulk_ce to phase diagram, if not present
    entries = phase_diagram.all_entries
    if not any(
        (ent.composition == bulk_ce.composition and ent.energy == bulk_ce.energy)
        for ent in entries
    ):
        entries.append(
            PDEntry(
                bulk_ce.composition,
                bulk_ce.energy,
                attribute="Bulk Material",
            )
        )
        phase_diagram = PhaseDiagram(entries)

    chem_lims = phase_diagram.get_all_chempots(redcomp)

    return chem_lims


class CompetingPhases:
    """
    Class to generate the input files for competing phases on the phase diagram for the host
    material (determining the chemical potential limits). Materials Project (MP) data is used,
    along with an uncertainty range specified by `e_above_hull`, to determine the relevant
    competing phases. Diatomic gaseous molecules are generated as molecules-in-a-box as appropriate.

    TODO: Need to add functionality to deal with cases where the bulk composition is not listed
    on the MP – warn user (i.e. check your sh*t) and generate the competing phases according to
    composition position within phase diagram.
    E.g. from pycdt chemical_potentials:
    #                 "However, no stable entry with this composition exists "
    #                 "in the MP database!\nPlease consider submitting the "
    #                 "POSCAR to the MP xtaltoolkit, so future users will "
    #                 "know about this structure:"
    #                 " https://materialsproject.org/#apps/xtaltoolkit\n" – see
    analyze_GGA_chempots code for example.
    TODO: Add note to notebook that if your bulk phase is lower energy than its version on the MP
    (e.g. distorted perovskite), then you should use this for your bulk competing phase calculation.
    """

    def __init__(
        self, composition, e_above_hull=0.1, api_key=None, full_phase_diagram=False
    ):
        """
        Args:
            composition (str, Composition): Composition of host material
                (e.g. 'LiFePO4', or Composition('LiFePO4'), or Composition({"Li":1, "Fe":1,
                "P":1, "O":4}))
            e_above_hull (float): Maximum energy-above-hull of Materials Project entries to be
                considered as competing phases. This is an uncertainty range for the
                MP-calculated formation energies, which may not be accurate due to functional
                choice (GGA vs hybrid DFT / GGA+U / RPA etc.), lack of vdW corrections etc.
                Any phases that (would) border the host material on the phase diagram, if their
                relative energy was downshifted by `e_above_hull`, are included.
                Default is 0.1 eV/atom.
            api_key (str): Materials Project (MP) API key, needed to access the MP database for
                competing phase generation. If not supplied, will attempt to read from
                environment variable `PMG_MAPI_KEY` (in `~/.pmgrc.yaml`) – see the `doped`
                homepage (https://github.com/SMTG-UCL/doped) for instructions on setting this up.
                This should correspond to the legacy MP API; from
                https://legacy.materialsproject.org/open.
            full_phase_diagram (bool): If True, include all phases on the MP phase diagram (
                with energy above hull < e_above_hull) for the chemical system of the input
                composition (not recommended). If False, only includes phases that (would) border
                the host material on the phase diagram (and thus set the chemical potential
                limits), if their relative energy was downshifted by `e_above_hull`.
                (Default is False).
        """
        self.api_key = api_key

        # create list of entries
        self._molecules_in_a_box = ["H2", "O2", "N2", "F2", "Cl2"]

        # TODO: Should hard code S (solid + S8), P, Te and Se in here too. Common anions with a
        #  lot of unnecessary polymorphs on MP
        # P_EaH=0 is red phosphorus (HSE06 groundstate), P_EaH=0.037 is black phosphorus (thermo
        # stable at RT), so only need to generate these

        # all data collected from materials project
        self.data = [
            "pretty_formula",
            "e_above_hull",
            "band_gap",
            "nsites",
            "volume",
            "icsd_id",
            "formation_energy_per_atom",
            "energy_per_atom",
            "energy",
            "total_magnetization",
            "nelements",
            "elements",
        ]

        # set bulk composition (Composition(Composition("LiFePO4")) = Composition("LiFePO4")))
        self.bulk_comp = Composition(composition)

        # test api_key:
        if api_key is not None:
            if len(api_key) == 32:
                raise ValueError(
                    "You are trying to use the new Materials Project (MP) API which is not "
                    "supported by doped. Please use the legacy MP API ("
                    "https://legacy.materialsproject.org/open)."
                )
            if 15 <= len(api_key) <= 20:
                self.eah = "e_above_hull"
            else:
                raise ValueError(
                    f"API key {api_key} is not a valid legacy Materials Project API key. These "
                    f"are available at https://legacy.materialsproject.org/open"
                )

        with MPRester(api_key=self.api_key) as mpr:
            self.MP_full_pd_entries = mpr.get_entries_in_chemsys(
                list(self.bulk_comp.as_dict().keys()),
                inc_structure="initial",
                property_data=self.data,
            )
        self.MP_full_pd_entries = [
            e for e in self.MP_full_pd_entries if e.data["e_above_hull"] <= e_above_hull
        ]
        self.MP_full_pd_entries.sort(
            key=lambda x: x.data["e_above_hull"]
        )  # sort by e_above_hull

        pd_entries = []
        # check that none of the elemental ones are molecules in a box
        for entry in self.MP_full_pd_entries.copy():
            if (
                entry.data["pretty_formula"] in self._molecules_in_a_box
                and entry.data["e_above_hull"] == 0
            ):
                # only first matching molecular entry
                molecular_entry = _make_molecular_entry(entry)
                if not any(
                    ent.data["molecule"]
                    and ent.data["pretty_formula"]
                    == molecular_entry.data["pretty_formula"]
                    for ent in pd_entries
                ):  # first entry only
                    pd_entries.append(molecular_entry)
                    self.MP_full_pd_entries.append(molecular_entry)

            elif entry.data["pretty_formula"] not in self._molecules_in_a_box:
                entry.data["molecule"] = False
                pd_entries.append(entry)

        pd_entries.sort(key=lambda x: x.energy_per_atom)  # sort by energy per atom
        phase_diagram = PhaseDiagram(pd_entries)
        bulk_entries = [
            entry
            for entry in pd_entries
            if entry.composition.reduced_composition
            == self.bulk_comp.reduced_composition
        ]
        bulk_ce = bulk_entries[
            0
        ]  # lowest energy entry for bulk composition (after sorting)
        self.MP_bulk_ce = bulk_ce

        if not full_phase_diagram:  # default
            # cull to only include any phases that would border the host material on the phase
            # diagram, if their relative energy was downshifted by `e_above_hull`:
            MP_gga_chempots = get_chempots_from_phase_diagram(bulk_ce, phase_diagram)

            MP_bordering_phases = {
                phase for facet in MP_gga_chempots for phase in facet.split("-")
            }
            self.entries = [
                entry
                for entry in pd_entries
                if entry.name in MP_bordering_phases or entry.is_element
            ]

            # add any phases that would border the host material on the phase diagram, if their
            # relative energy was downshifted by `e_above_hull`:
            for entry in pd_entries:
                if entry.name not in MP_bordering_phases and not entry.is_element:
                    # decrease entry energy per atom by `e_above_hull` eV/atom
                    renormalised_entry = _renormalise_entry(entry, e_above_hull)
                    new_phase_diagram = PhaseDiagram(
                        phase_diagram.entries + [renormalised_entry]
                    )
                    new_MP_gga_chempots = get_chempots_from_phase_diagram(
                        bulk_ce, new_phase_diagram
                    )

                    if new_MP_gga_chempots != MP_gga_chempots:
                        # new bordering phase, add to list
                        self.entries.append(entry)

        else:  # full_phase_diagram = True
            self.entries = pd_entries

        # sort entries by e_above_hull, and then by num_species, then alphabetically:
        self.entries.sort(
            key=lambda x: (
                x.data["e_above_hull"],
                len(Composition(x.name).as_dict()),
                x.name,
            )
        )

    # TODO: Similar refactor to work mainly off config dict object here as well (as vasp_input)?
    # TODO: Currently doesn't exactly match vaspup2.0 naming convention which means it doesn't
    #  account for the ordering switch from 1..9 to 10 etc
    def convergence_setup(
        self,
        kpoints_metals=(40, 120, 5),
        kpoints_nonmetals=(5, 80, 5),
        user_potcar_functional="PBE_54",
        user_potcar_settings=None,
        user_incar_settings=None,
        **kwargs,
    ):
        """
        Generates VASP input files for kpoints convergence testing for competing phases,
        using PBEsol (GGA) DFT by default. Automatically sets the `ISMEAR` `INCAR` tag to 2 (if
        metallic) or 0 if not. Recommended to use with https://github.com/kavanase/vaspup2.0.

        Args:
            kpoints_metals (tuple): Kpoint density per inverse volume (Å^-3) to be tested in
                (min, max, step) format for metals
            kpoints_nonmetals (tuple): Kpoint density per inverse volume (Å^-3) to be tested in
                (min, max, step) format for nonmetals
            user_potcar_functional (str): POTCAR functional to use (default = "PBE_54")
            user_potcar_settings (dict): Override the default POTCARs, e.g. {"Li": "Li_sv"}. See
                `doped/PotcarSet.yaml` for the default `POTCAR` set.
            user_incar_settings (dict): Override the default INCAR settings
                e.g. {"EDIFF": 1e-5, "LDAU": False, "ALGO": "All"}. Note that any non-numerical or
                non-True/False flags need to be input as strings with quotation marks. See
                `doped/PBEsol_ConvergenceSet.yaml` for the default settings.
        """
        # by default uses pbesol, but easy to switch to pbe or pbe+u using user_incar_settings
        pbesol_convrg_set = loadfn(
            os.path.join(MODULE_DIR, "PBEsol_ConvergenceSet.yaml")
        )

        # kpoints should be set as (min, max, step)
        min_nm, max_nm, step_nm = kpoints_nonmetals
        min_m, max_m, step_m = kpoints_metals

        potcar_dict = copy.deepcopy(default_potcar_dict)
        if user_potcar_settings:
            potcar_dict["POTCAR"].update(user_potcar_settings)
        if user_potcar_functional:
            potcar_dict["POTCAR_FUNCTIONAL"] = user_potcar_functional
        pbesol_convrg_set.update(potcar_dict)

        # separate metals and non-metals
        self.nonmetals = []
        self.metals = []
        for e in self.entries:
            if not e.data["molecule"]:
                if e.data["band_gap"] > 0:
                    self.nonmetals.append(e)
                else:
                    self.metals.append(e)
            else:
                print(
                    f"{e.name} is a molecule in a box, does not need convergence testing"
                )

        for e in self.nonmetals:
            if user_incar_settings is not None:
                uis = copy.deepcopy(user_incar_settings)
            else:
                uis = {}
            if e.data["total_magnetization"] > 0.1:  # account for magnetic moment
                if "ISPIN" not in uis:
                    uis["ISPIN"] = 2
                if "NUPDOWN" not in uis and int(e.data["total_magnetization"]) > 0:
                    uis["NUPDOWN"] = int(e.data["total_magnetization"])

            for kpoint in range(min_nm, max_nm, step_nm):
                dis = DictSet(
                    e.structure,
                    pbesol_convrg_set,
                    user_kpoints_settings={"reciprocal_density": kpoint},
                    user_incar_settings=uis,
                    force_gamma=True,
                )

                kname = "k" + ",".join(str(k) for k in dis.kpoints.kpts[0])
                fname = (
                    f"competing_phases/{e.name}_EaH"
                    f"_{round(e.data['e_above_hull'],4)}/kpoint_converge/{kname}"
                )  # TODO: competing_phases folder name should be an optional parameter
                dis.write_input(fname, **kwargs)

        for e in self.metals:
            if user_incar_settings is not None:
                uis = copy.deepcopy(user_incar_settings)
            else:
                uis = {}
            # change the ismear and sigma for metals
            uis["ISMEAR"] = -5
            uis["SIGMA"] = 0.2

            if e.data["total_magnetization"] > 0.1:  # account for magnetic moment
                if "ISPIN" not in uis:
                    uis["ISPIN"] = 2
                if "NUPDOWN" not in uis and int(e.data["total_magnetization"]) > 0:
                    uis["NUPDOWN"] = int(e.data["total_magnetization"])

            for kpoint in range(min_m, max_m, step_m):
                dis = DictSet(
                    e.structure,
                    pbesol_convrg_set,
                    user_kpoints_settings={"reciprocal_density": kpoint},
                    user_incar_settings=uis,
                    force_gamma=True,
                )

                kname = "k" + ",".join(str(k) for k in dis.kpoints.kpts[0])
                fname = (
                    f"competing_phases/{e.name}_EaH_"
                    f"{round(e.data['e_above_hull'],4)}/kpoint_converge/{kname}"
                )
                dis.write_input(fname, **kwargs)

    # TODO: Add vasp_ncl_setup()
    def vasp_std_setup(
        self,
        kpoints_metals=95,
        kpoints_nonmetals=45,
        user_potcar_functional="PBE_54",
        user_potcar_settings=None,
        user_incar_settings=None,
        **kwargs,
    ):
        """
        Generates VASP input files for `vasp_std` relaxations of the competing phases,
        using HSE06 (hybrid DFT) DFT by default. Automatically sets the `ISMEAR` `INCAR` tag to 2
        (if metallic) or 0 if not. Note that any changes to the default `INCAR`/`POTCAR` settings
        should be consistent with those used for the defect supercell calculations.

        Args:
            kpoints_metals (int): Kpoint density per inverse volume (Å^-3) for metals
            kpoints_nonmetals (int): Kpoint density per inverse volume (Å^-3) for nonmetals
            user_potcar_functional (str): POTCAR functional to use (default = "PBE_54")
            user_potcar_settings (dict): Override the default POTCARs, e.g. {"Li": "Li_sv"}. See
                `doped/PotcarSet.yaml` for the default `POTCAR` set.
            user_incar_settings (dict): Override the default INCAR settings
                e.g. {"EDIFF": 1e-5, "LDAU": False, "ALGO": "All"}. Note that any non-numerical or
                non-True/False flags need to be input as strings with quotation marks. See
                `doped/HSE06_RelaxSet.yaml` for the default settings.
        """
        # TODO: Update this to use:
        #  sym = SpacegroupAnalyzer(e.structure)
        #  struc = sym.get_primitive_standard_structure() -> output this structure
        hse06_relax_set = loadfn(os.path.join(MODULE_DIR, "HSE06_RelaxSet.yaml"))

        potcar_dict = copy.deepcopy(default_potcar_dict)
        if user_potcar_settings:
            potcar_dict["POTCAR"].update(user_potcar_settings)
        if user_potcar_functional:
            potcar_dict["POTCAR_FUNCTIONAL"] = user_potcar_functional
        hse06_relax_set.update(potcar_dict)

        # separate metals, non-metals and molecules
        self.nonmetals = []
        self.metals = []
        self.molecules = []
        for e in self.entries:
            if e.data["molecule"]:
                self.molecules.append(e)
            else:
                if e.data["band_gap"] > 0:
                    self.nonmetals.append(e)
                else:
                    self.metals.append(e)

        for e in self.nonmetals:
            if user_incar_settings is not None:
                uis = copy.deepcopy(user_incar_settings)
            else:
                uis = {}
            if e.data["total_magnetization"] > 0.1:  # account for magnetic moment
                if "ISPIN" not in uis:
                    uis["ISPIN"] = 2
                if "NUPDOWN" not in uis and int(e.data["total_magnetization"]) > 0:
                    uis["NUPDOWN"] = int(e.data["total_magnetization"])

            dis = DictSet(
                e.structure,
                hse06_relax_set,
                user_kpoints_settings={"reciprocal_density": kpoints_nonmetals},
                user_incar_settings=uis,
                force_gamma=True,
            )

            fname = (
                f"competing_phases/{e.name}_EaH_"
                f"{round(e.data['e_above_hull'],4)}/vasp_std"
            )
            dis.write_input(fname, **kwargs)

        for e in self.metals:
            if user_incar_settings is not None:
                uis = copy.deepcopy(user_incar_settings)
            else:
                uis = {}
            # change the ismear and sigma for metals
            uis["ISMEAR"] = 1
            uis["SIGMA"] = 0.2

            if e.data["total_magnetization"] > 0.1:  # account for magnetic moment
                if "ISPIN" not in uis:
                    uis["ISPIN"] = 2
                if "NUPDOWN" not in uis and int(e.data["total_magnetization"]) > 0:
                    uis["NUPDOWN"] = int(e.data["total_magnetization"])

            dis = DictSet(
                e.structure,
                hse06_relax_set,
                user_kpoints_settings={"reciprocal_density": kpoints_metals},
                user_incar_settings=uis,
                force_gamma=True,
            )
            fname = (
                f"competing_phases/{e.name}_EaH_"
                f"{round(e.data['e_above_hull'],4)}/vasp_std"
            )
            dis.write_input(fname, **kwargs)

        for e in self.molecules:  # gamma-only for molecules
            if user_incar_settings is not None:
                uis = copy.deepcopy(user_incar_settings)
            else:
                uis = {}

            uis["ISIF"] = 2  # can't change the volume

            if e.data["total_magnetization"] > 0.1:  # account for magnetic moment
                if "ISPIN" not in uis:
                    uis["ISPIN"] = 2
                if "NUPDOWN" not in uis and int(e.data["total_magnetization"]) > 0:
                    uis["NUPDOWN"] = int(e.data["total_magnetization"])

            dis = DictSet(
                e.structure,
                hse06_relax_set,
                user_kpoints_settings=Kpoints().from_dict(
                    {
                        "comment": "Gamma-only kpoints for molecule-in-a-box",
                        "generation_style": "Gamma",
                    }
                ),
                user_incar_settings=uis,
                force_gamma=True,
            )
            fname = f"competing_phases/{e.name}_EaH_{round(e.data['e_above_hull'],4)}/vasp_std"
            dis.write_input(fname, **kwargs)


# TODO: Add full_sub_approach option
# TODO: Add warnings for full_sub_approach=True, especially if done with multiple
#  extrinsic species.
class ExtrinsicCompetingPhases(CompetingPhases):
    """
    This class generates the competing phases that need to be calculated to obtain the chemical
    potential limits when doping with extrinsic species / impurities. Ensures that only the
    necessary additional competing phases are generated.
    """

    def __init__(
        self,
        composition,
        extrinsic_species,
        e_above_hull=0.1,
        full_sub_approach=False,
        codoping=False,
        api_key=None,
    ):
        """
        Args:
            composition (str, Composition): Composition of host material
                (e.g. 'LiFePO4', or Composition('LiFePO4'), or Composition({"Li":1, "Fe":1,
                "P":1, "O":4}))
            extrinsic_species (str, list): Extrinsic dopant/impurity species
                (e.g. "Mg" or ["Mg", "Na"])
            e_above_hull (float): Maximum energy-above-hull of Materials Project entries to be
                considered as competing phases. This is an uncertainty range for the
                MP-calculated formation energies, which may not be accurate due to functional
                choice (GGA vs hybrid DFT / GGA+U / RPA etc.), lack of vdW corrections etc.
                Any phases that would border the host material on the phase diagram, if their
                relative energy was downshifted by `e_above_hull`, are included.
                Default is 0.1 eV/atom.
            full_sub_approach (bool): Generate competing phases by considering the full phase
                diagram, including chemical potential limits with multiple extrinsic phases.
                Only recommended when looking at high (non-dilute) doping concentrations.
                Default = False. Described in further detail below.
            codoping (bool): Whether to consider extrinsic competing phases containing multiple
                extrinsic species. Only relevant to high (non-dilute) co-doping concentrations.
                If set to True, then `full_sub_approach` is also set to True.
                Default = False.
            api_key (str): Materials Project (MP) API key, needed to access the MP database for
                competing phase generation. If not supplied, will attempt to read from
                environment variable `PMG_MAPI_KEY` (in `~/.pmgrc.yaml`) – see the `doped`
                homepage (https://github.com/SMTG-UCL/doped) for instructions on setting this up.
                This should correspond to the legacy MP API; from
                https://legacy.materialsproject.org/open.

        This code uses the Materials Project (MP) phase diagram data along with the
        `e_above_hull` error range to generate potential competing phases.

        NOTE on 'full_sub_approach':
            The default approach for substitutional elements (`full_sub_approach = False`) is to
            only consider chemical potential limits with a maximum of 1 extrinsic phase
            (composition with extrinsic species present). This is a valid approximation for the
            case of dilute dopant/impurity concentrations. For high (non-dilute) concentrations
            of extrinsic species, use `full_sub_approach = True`.
        """
        # the competing phases & entries of the OG system
        super().__init__(composition, e_above_hull, api_key)
        self.intrinsic_entries = copy.deepcopy(self.entries)
        self.entries = []
        self.intrinsic_species = [
            s.symbol for s in self.bulk_comp.reduced_composition.elements
        ]
        self.MP_intrinsic_full_pd_entries = (
            self.MP_full_pd_entries
        )  # includes molecules-in-boxes

        if isinstance(extrinsic_species, str):
            extrinsic_species = [
                extrinsic_species,
            ]
        elif not isinstance(extrinsic_species, list):
            raise TypeError(
                f"`extrinsic_species` must be a string (i.e. the extrinsic species "
                f"symbol, e.g. 'Mg') or a list (e.g. ['Mg', 'Na']), got type "
                f"{type(extrinsic_species)} instead!"
            )
        self.extrinsic_species = extrinsic_species

        # if codoping = True, should have multiple extrinsic species
        if codoping:
            if len(extrinsic_species) < 2:
                warnings.warn(
                    "`codoping` is set to True, but `extrinsic_species` only contains 1 "
                    "element, so `codoping` will be set to False."
                )
                codoping = False

            elif not full_sub_approach:
                full_sub_approach = True

        if (
            full_sub_approach
        ):  # can be time-consuming if several extrinsic_species supplied
            if codoping:
                # TODO: `full_sub_approach` shouldn't necessarily mean `full_phase_diagram =
                #  True` right? As in can be non-full-phase-diagram intrinsic + extrinsic
                #  entries, including facets with multiple extrinsic entries but still not the
                #  full phase diagram? – To be updated!
                # TODO: When `full_phase_diagram` option added to `CompetingPhases`, can remove
                #  this code block and just use:
                #  super()__init__(composition = (
                #  self.intrinsic_species+self.extrinsic_species).join(""), e_above_hull, api_key)
                #  )
                #  self.intrinsic_entries = [phase for phase in self.entries if
                #  not any([extrinsic in phase for extrinsic in self.extrinsic_species])]
                #  entries = [phase for phase in self.entries if phase not in
                #  self.intrinsic_entries]
                #  self.entries = entries
                #  self.MP_intrinsic_full_pd_entries = [entry for entry in
                #  self.MP_full_pd_entries if not any([extrinsic in entry.composition.reduced_formula
                #  for extrinsic in self.extrinsic_species])]
                #  MP_full_pd_entries = [entry for entry in self.MP_full_pd_entries if entry not in
                #  self.MP_intrinsic_full_pd_entries]
                #  self.MP_full_pd_entries = MP_full_pd_entries  # includes molecules-in-boxes

                with MPRester(api_key=self.api_key) as mpr:
                    self.MP_full_pd_entries = mpr.get_entries_in_chemsys(
                        self.intrinsic_species + self.extrinsic_species,
                        inc_structure="initial",
                        property_data=self.data,
                    )
                self.MP_full_pd_entries = [
                    e
                    for e in self.MP_full_pd_entries
                    if e.data["e_above_hull"] <= e_above_hull
                ]

                # sort by e_above_hull:
                self.MP_full_pd_entries.sort(key=lambda x: x.data["e_above_hull"])

                for entry in self.MP_full_pd_entries.copy():
                    if any(
                        sub_elt in entry.composition
                        for sub_elt in self.extrinsic_species
                    ):
                        if (
                            entry.data["pretty_formula"] in self._molecules_in_a_box
                            and entry.data["e_above_hull"] == 0
                        ):  # only first matching entry
                            molecular_entry = _make_molecular_entry(entry)
                            if not any(
                                entry.data["molecule"]
                                and entry.data["pretty_formula"]
                                == molecular_entry.data["pretty_formula"]
                                for entry in self.entries
                            ):  # first entry only
                                self.MP_full_pd_entries.append(molecular_entry)
                                self.entries.append(molecular_entry)
                        elif (
                            entry.data["pretty_formula"] not in self._molecules_in_a_box
                        ):
                            entry.data["molecule"] = False
                            self.entries.append(entry)

                # sort entries by e_above_hull, and then by num_species, then alphabetically:
                self.entries.sort(
                    key=lambda x: (
                        x.data["e_above_hull"],
                        len(Composition(x.name).as_dict()),
                        x.name,
                    )
                )

            else:  # full_sub_approach but not co-doping
                self.MP_full_pd_entries = []
                for sub_elt in self.extrinsic_species:
                    extrinsic_entries = []
                    with MPRester(api_key=self.api_key) as mpr:
                        MP_full_pd_entries = mpr.get_entries_in_chemsys(
                            self.intrinsic_species
                            + [
                                sub_elt,
                            ],
                            inc_structure="initial",
                            property_data=self.data,
                        )
                    MP_full_pd_entries = [
                        e
                        for e in MP_full_pd_entries
                        if e.data["e_above_hull"] <= e_above_hull
                    ]
                    # sort by e_above_hull:
                    MP_full_pd_entries.sort(key=lambda x: x.data["e_above_hull"])

                    for entry in MP_full_pd_entries.copy():
                        if entry not in self.MP_full_pd_entries:
                            self.MP_full_pd_entries.append(entry)
                        if sub_elt in entry.composition:
                            if (
                                entry.data["pretty_formula"] in self._molecules_in_a_box
                                and entry.data["e_above_hull"] == 0
                            ):
                                # only first matching entry
                                molecular_entry = _make_molecular_entry(entry)
                                if not any(
                                    entry.data["molecule"]
                                    and entry.data["pretty_formula"]
                                    == molecular_entry.data["pretty_formula"]
                                    for entry in self.entries
                                ):  # first entry only
                                    self.MP_full_pd_entries.append(molecular_entry)
                                    extrinsic_entries.append(molecular_entry)

                            elif (
                                entry.data["pretty_formula"]
                                not in self._molecules_in_a_box
                            ):
                                entry.data["molecule"] = False
                                extrinsic_entries.append(entry)

                            # sort entries by e_above_hull, and then by num_species,
                            # then alphabetically:
                            extrinsic_entries.sort(
                                key=lambda x: (
                                    x.data["e_above_hull"],
                                    len(Composition(x.name).as_dict()),
                                    x.name,
                                )
                            )
                            self.entries += extrinsic_entries

        else:  # full_sub_approach = False; recommended approach for extrinsic species (assumes
            # dilute concentrations)

            # now compile substitution entries:
            self.MP_full_pd_entries = []

            for sub_elt in self.extrinsic_species:
                extrinsic_pd_entries = []
                with MPRester(api_key=self.api_key) as mpr:
                    MP_full_pd_entries = mpr.get_entries_in_chemsys(
                        self.intrinsic_species
                        + [
                            sub_elt,
                        ],
                        inc_structure="initial",
                        property_data=self.data,
                    )
                self.MP_full_pd_entries = [
                    e
                    for e in MP_full_pd_entries
                    if e.data["e_above_hull"] <= e_above_hull
                ]
                # sort by e_above_hull:
                self.MP_full_pd_entries.sort(key=lambda x: x.data["e_above_hull"])

                for entry in self.MP_full_pd_entries.copy():
                    if (
                        entry.data["pretty_formula"] in self._molecules_in_a_box
                        and entry.data["e_above_hull"] == 0
                    ):
                        # only first matching entry
                        molecular_entry = _make_molecular_entry(entry)
                        if not any(
                            entry.data["molecule"]
                            and entry.data["pretty_formula"]
                            == molecular_entry.data["pretty_formula"]
                            for entry in extrinsic_pd_entries
                        ):  # first entry only
                            self.MP_full_pd_entries.append(molecular_entry)
                            extrinsic_pd_entries.append(molecular_entry)
                    elif entry.data["pretty_formula"] not in self._molecules_in_a_box:
                        entry.data["molecule"] = False
                        extrinsic_pd_entries.append(entry)

                # Adding substitutional phases to extrinsic competing phases list only when the
                # phases in equilibria are those from the bulk phase diagram. This is essentially
                # the assumption that the majority of elements in the total composition will be
                # from the host composition rather than the extrinsic species (a good
                # approximation for dilute concentrations)

                if not extrinsic_pd_entries:
                    raise ValueError(
                        f"No Materials Project entries found for the given chemical "
                        f"system: {self.intrinsic_species + [sub_elt,]}"
                    )

                extrinsic_phase_diagram = PhaseDiagram(extrinsic_pd_entries)
                MP_extrinsic_gga_chempots = get_chempots_from_phase_diagram(
                    self.MP_bulk_ce, extrinsic_phase_diagram
                )
                MP_extrinsic_bordering_phases = []

                for facet in MP_extrinsic_gga_chempots.keys():
                    # if the number of intrinsic competing phases for this facet is equal to the
                    # number of species in the bulk composition, then include the extrinsic phase(s)
                    # for this facet (full_sub_approach = False approach)
                    MP_intrinsic_bordering_phases = {
                        phase for phase in facet.split("-") if sub_elt not in phase
                    }
                    if len(MP_intrinsic_bordering_phases) == len(
                        self.intrinsic_species
                    ):
                        # add to list of extrinsic bordering phases, if not already present:
                        MP_extrinsic_bordering_phases.extend(
                            [
                                phase
                                for phase in facet.split("-")
                                if sub_elt in phase
                                and phase not in MP_extrinsic_bordering_phases
                            ]
                        )

                # add any phases that would border the host material on the phase diagram,
                # if their relative energy was downshifted by `e_above_hull`:
                for entry in MP_full_pd_entries:
                    if (
                        entry.name not in MP_extrinsic_bordering_phases
                        and not entry.is_element
                        and sub_elt in entry.composition
                    ):
                        # decrease entry energy per atom by `e_above_hull` eV/atom
                        renormalised_entry = _renormalise_entry(entry, e_above_hull)
                        new_extrinsic_phase_diagram = PhaseDiagram(
                            extrinsic_phase_diagram.entries + [renormalised_entry]
                        )
                        new_MP_extrinsic_gga_chempots = get_chempots_from_phase_diagram(
                            self.MP_bulk_ce, new_extrinsic_phase_diagram
                        )

                        if new_MP_extrinsic_gga_chempots != MP_extrinsic_gga_chempots:
                            # new bordering phase, check if not an over-dependent facet:

                            for facet in new_MP_extrinsic_gga_chempots:
                                if facet not in MP_extrinsic_gga_chempots:
                                    # new facet, check if not an over-dependent facet:
                                    MP_intrinsic_bordering_phases = {
                                        phase
                                        for phase in facet.split("-")
                                        if sub_elt not in phase
                                    }
                                    if len(MP_intrinsic_bordering_phases) == len(
                                        self.intrinsic_species
                                    ):
                                        MP_extrinsic_bordering_phases.extend(
                                            [
                                                phase
                                                for phase in facet.split("-")
                                                if sub_elt in phase
                                                and phase
                                                not in MP_extrinsic_bordering_phases
                                            ]
                                        )

                extrinsic_entries = [
                    entry
                    for entry in extrinsic_pd_entries
                    if entry.name in MP_extrinsic_bordering_phases
                    or (entry.is_element and sub_elt in entry.name)
                ]

                # check that extrinsic competing phases list is not empty (can happen with
                # 'over-dependent' facets); if so then set full_sub_approach = True and re-run
                # the extrinsic phase addition process
                if not extrinsic_entries:
                    warnings.warn(
                        "Determined chemical potentials to be over dependent on an "
                        "extrinsic species. This means we need to revert to "
                        "`full_sub_approach = True` – running now."
                    )
                    full_sub_approach = True
                    extrinsic_entries = [
                        entry
                        for entry in self.MP_full_pd_entries
                        if sub_elt in entry.composition
                    ]

                # sort entries by e_above_hull, and then by num_species, then alphabetically:
                extrinsic_entries.sort(
                    key=lambda x: (
                        x.data["e_above_hull"],
                        len(Composition(x.name).as_dict()),
                        x.name,
                    )
                )
                self.entries += extrinsic_entries


class CompetingPhasesAnalyzer:
    """
    Post processing competing phases data to calculate chemical potentials.
    """

    def __init__(self, system, extrinsic_species=None):
        """
        Args:
            system (str): The  'reduced formula' of the bulk composition
            extrinsic_species (str): Dopant species - can only deal with one at a time (see
            notebook in examples folder for more complex cases)
        """

        self.bulk_composition = Composition(system)
        self.elemental = [str(c) for c in self.bulk_composition.elements]
        self.extrinsic_species = extrinsic_species

        if extrinsic_species:
            self.elemental.append(extrinsic_species)

    def from_vaspruns(self, path="competing_phases", folder="vasp_std", csv_fname=None):
        """
        Reads in vaspruns, collates energies to csv.

        Args:
            path (list, str, pathlib Path): Either a list of strings or Paths to vasprun.xml(.gz)
            files, or a path to the base folder in which you have your
            formula_EaH_/vasp_std/vasprun.xml
            folder (str): The folder in which vasprun is, only use if you set base path
            (i.e. change to vasp_ncl or if vaspruns are in the formula_EaH folder).
            csv_fname (str): If set will save to csv with this name
        Returns:
            None, sets self.data and self.elemental_energies
        """
        # TODO: Change this to just recursively search for vaspruns within the specified path
        # TODO: Add check for matching INCAR and POTCARs from these calcs, as we also want with
        #  defect parsing
        self.vasprun_paths = []
        # fetch data
        # if path is just a list of all competing phases
        if isinstance(path, list):
            for p in path:
                if "vasprun.xml" in Path(p).name:
                    self.vasprun_paths.append(str(Path(p)))

                # try to find the file - will always pick the first match for vasprun.xml*
                elif len(list(Path(p).glob("vasprun.xml*"))) > 0:
                    vsp = list(Path(p).glob("vasprun.xml*"))[0]
                    self.vasprun_paths.append(str(vsp))

                else:
                    print(
                        f"Can't find a vasprun.xml(.gz) file for {p}, proceed with caution"
                    )

        # if path provided points to the doped created directories
        elif isinstance(path, (PurePath, str)):
            path = Path(path)
            for p in path.iterdir():
                if p.glob("EaH"):
                    vp = p / folder / "vasprun.xml"
                    try:
                        vr, vr_path = get_vasprun(vp)
                        self.vasprun_paths.append(vr_path)

                    except FileNotFoundError:
                        try:
                            vp = p / "vasprun.xml"
                            vr, vr_path = get_vasprun(vp)
                            self.vasprun_paths.append(vr_path)

                        except FileNotFoundError:
                            print(
                                f"Can't find a vasprun.xml(.gz) file in {p} or {p/folder}, "
                                f"proceed with caution"
                            )
                            continue

                else:
                    raise FileNotFoundError(
                        "Folders are not in the correct structure, provide them as a list of "
                        "paths (or strings)"
                    )

        else:
            raise ValueError(
                "Path should either be a list of paths, a string or a pathlib Path object"
            )

        # Ignore POTCAR warnings when loading vasprun.xml
        # pymatgen assumes the default PBE with no way of changing this
        warnings.filterwarnings("ignore", category=UnknownPotcarWarning)
        warnings.filterwarnings(
            "ignore", message="No POTCAR file with matching TITEL fields"
        )

        num = len(self.vasprun_paths)
        print(
            f"Parsing {num} vaspruns and pruning to include only lowest-energy polymorphs..."
        )
        self.vaspruns = [Vasprun(e).as_dict() for e in self.vasprun_paths]
        self.data = []

        temp_data = []
        self.elemental_energies = {}
        for v in self.vaspruns:
            rcf = v["reduced_cell_formula"]
            comp = Composition(v["unit_cell_formula"])
            formulas_per_unit = comp.get_reduced_composition_and_factor()[1]
            final_energy = v["output"]["final_energy"]
            kpoints = "x".join(str(x) for x in v["input"]["kpoints"]["kpoints"][0])

            # check if elemental:
            if len(rcf) == 1:
                elt = v["elements"][0]
                if elt not in self.elemental_energies:
                    self.elemental_energies[elt] = v["output"]["final_energy_per_atom"]
                    if elt not in self.elemental:  # new (extrinsic) element
                        self.extrinsic_species = elt
                        self.elemental.append(elt)

                elif (
                    v["output"]["final_energy_per_atom"] < self.elemental_energies[elt]
                ):
                    # only include lowest energy elemental polymorph
                    self.elemental_energies[elt] = v["output"]["final_energy_per_atom"]

            d = {
                "formula": v["pretty_formula"],
                "kpoints": kpoints,
                "energy_per_fu": final_energy / formulas_per_unit,
                "energy_per_atom": v["output"]["final_energy_per_atom"],
                "energy": final_energy,
            }
            temp_data.append(d)

        df = _calculate_formation_energies(temp_data, self.elemental_energies)
        if csv_fname is not None:
            df.to_csv(csv_fname, index=False)
            print(f"Competing phase formation energies have been saved to {csv_fname}.")

        self.data = df.to_dict(orient="records")

    def from_csv(self, csv):
        """
        Read in data from csv.
        Args:
            csv (str): Path to csv file. Must have columns 'formula',
                'energy_per_fu', 'energy' and 'formation_energy'
        Returns:
            None, sets self.data and self.elemental_energies
        """
        df = pd.read_csv(csv)
        columns = [
            "formula",
            "energy_per_fu",
            "energy_per_atom",
            "energy",
            "formation_energy",
        ]
        if all(x in list(df.columns) for x in columns):
            droplist = [i for i in df.columns if i not in columns]
            df.drop(droplist, axis=1, inplace=True)
            d = df.to_dict(orient="records")
            self.data = d

            self.elemental_energies = {}
            for i in d:
                c = Composition(i["formula"])
                if len(c.elements) == 1:
                    elt = c.chemical_system
                    if (
                        elt not in self.elemental_energies
                        or i["energy_per_atom"] < self.elemental_energies[elt]
                    ):
                        self.elemental_energies[elt] = i["energy_per_atom"]

        else:
            raise ValueError(
                "supplied csv does not contain the correct headers, cannot read in the data"
            )

    def calculate_chempots(self, csv_fname=None, verbose=True):
        """
        Calculates chemcial potential limits. For dopant species, it calculates the limiting
        potential based on the intrinsic chemical potentials (i.e. same as
        `full_sub_approach=False` in pycdt)
        Args:
            csv_fname (str): If set, will save chemical potential limits to csv
            verbose (bool): If True, will print out chemical potential limits.
        Retruns:
            Pandas DataFrame, optionally saved to csv
        """

        intrinsic_phase_diagram_entries = []
        extrinsic_formation_energies = []
        bulk_pde_list = []
        for d in self.data:
            e = PDEntry(d["formula"], d["energy_per_fu"])
            # presumably checks if the phase is intrinsic
            if set(Composition(d["formula"]).elements).issubset(
                self.bulk_composition.elements
            ):
                intrinsic_phase_diagram_entries.append(e)
                if e.composition == self.bulk_composition:  # bulk phase
                    bulk_pde_list.append(e)
            else:
                extrinsic_formation_energies.append(
                    {"formula": d["formula"], "formation_energy": d["formation_energy"]}
                )

        if len(bulk_pde_list) == 0:
            raise ValueError(
                f"Could not find bulk phase for "
                f"{self.bulk_composition.reduced_formula} in the supplied data. "
                f"Found phases: "
                f"{ {e.composition.reduced_formula for e in intrinsic_phase_diagram_entries} }"
            )
        if len(bulk_pde_list) > 0:
            # lowest energy bulk phase
            self.bulk_pde = sorted(bulk_pde_list, key=lambda x: x.energy_per_atom)[0]

        self._intrinsic_phase_diagram = PhaseDiagram(
            intrinsic_phase_diagram_entries,
            map(Element, self.bulk_composition.elements),
        )

        # check if it's stable and if not error out
        if self.bulk_pde not in self._intrinsic_phase_diagram.stable_entries:
            eah = self._intrinsic_phase_diagram.get_e_above_hull(self.bulk_pde)
            raise ValueError(
                f"{self.bulk_composition.reduced_formula} is not stable with respect to competing "
                f"phases, EaH={eah:.4f} eV/atom"
            )

        chem_lims = self._intrinsic_phase_diagram.get_all_chempots(
            self.bulk_composition
        )
        # remove Element to make it jsonable
        no_element_chem_lims = {}
        for k, v in chem_lims.items():
            temp_dict = {}
            for kk, vv in v.items():
                temp_dict[str(kk)] = vv
            no_element_chem_lims[k] = temp_dict

        self._intrinsic_chem_limits = {
            "facets": no_element_chem_lims,
            "elemental_refs": {
                str(elt): ent.energy_per_atom
                for elt, ent in self._intrinsic_phase_diagram.el_refs.items()
            },
            "facets_wrt_el_refs": {},
        }

        # relate the facets to the elemental energies
        for facet, chempot_dict in self._intrinsic_chem_limits["facets"].items():
            relative_chempot_dict = copy.deepcopy(chempot_dict)
            for e in relative_chempot_dict.keys():
                relative_chempot_dict[e] -= self._intrinsic_chem_limits[
                    "elemental_refs"
                ][e]
            self._intrinsic_chem_limits["facets_wrt_el_refs"].update(
                {facet: relative_chempot_dict}
            )

        # get chemical potentials as pandas dataframe
        chemical_potentials = []
        for k, v in self._intrinsic_chem_limits["facets_wrt_el_refs"].items():
            lst = []
            columns = []
            for k, v in v.items():
                lst.append(v)
                columns.append(str(k))
            chemical_potentials.append(lst)

        # make df, will need it in next step
        df = pd.DataFrame(chemical_potentials, columns=columns)

        if self.extrinsic_species is not None:
            if verbose:
                print(f"Calculating chempots for {self.extrinsic_species}")
            for e in extrinsic_formation_energies:
                for el in self.elemental:
                    e[el] = Composition(e["formula"]).as_dict()[el]

            # gets the df into a slightly more convenient dict
            cpd = df.to_dict(orient="records")
            mins = []
            mins_formulas = []
            df3 = pd.DataFrame(extrinsic_formation_energies)
            for i, c in enumerate(cpd):
                name = f"mu_{self.extrinsic_species}_{i}"
                df3[name] = df3["formation_energy"]
                for k, v in c.items():
                    df3[name] -= df3[k] * v
                df3[name] /= df3[self.extrinsic_species]
                # find min at that chempot
                mins.append(df3[name].min())
                mins_formulas.append(df3.iloc[df3[name].idxmin()]["formula"])

            df[self.extrinsic_species] = mins
            col_name = f"{self.extrinsic_species}_limiting_phase"
            df[col_name] = mins_formulas

            # 1. work out the formation energies of all dopant competing
            #    phases using the elemental energies
            # 2. for each of the chempots already calculated work out what
            #    the chemical potential of the dopant would be from
            #       mu_dopant = Hf(dopant competing phase) - sum(mu_elements)
            # 3. find the most negative mu_dopant which then becomes the new
            #    canonical chemical potential for that dopant species and the
            #    competing phase is the 'limiting phase' right?
            # 4. update the chemical potential limits table to reflect this?
            # TODO: I don't think this is right. Here it's just getting the extrinsic chempots at
            #  the intrinsic chempot limits, but actually it should be recalculating the chempot
            #  limits with the extrinsic competing phases, then reducing _these_ facets down to
            #  those with only one extrinsic competing phase bordering

            # reverse engineer chem lims for extrinsic
            df4 = df.copy().to_dict(orient="records")
            cl2 = {
                "elemental_refs": self.elemental_energies,
                "facets_wrt_el_refs": {},
                "facets": {},
            }

            for i, d in enumerate(df4):
                key = (
                    list(self._intrinsic_chem_limits["facets_wrt_el_refs"].keys())[i]
                    + "-"
                    + d[col_name]
                )
                new_vals = list(
                    self._intrinsic_chem_limits["facets_wrt_el_refs"].values()
                )[i]
                new_vals[f"{self.extrinsic_species}"] = d[f"{self.extrinsic_species}"]
                cl2["facets_wrt_el_refs"][key] = new_vals

            # relate the facets to the elemental
            # energies but in reverse this time
            for facet, chempot_dict in cl2["facets_wrt_el_refs"].items():
                relative_chempot_dict = copy.deepcopy(chempot_dict)
                for e in relative_chempot_dict.keys():
                    relative_chempot_dict[e] += cl2["elemental_refs"][e]
                cl2["facets"].update({facet: relative_chempot_dict})

            self._chem_limits = cl2

        else:  # intrinsic only
            self._chem_limits = self._intrinsic_chem_limits

        # save and print
        if csv_fname is not None:
            df.to_csv(csv_fname, index=False)
            if verbose:
                print("Saved chemical potential limits to csv file: ", csv_fname)

        if verbose:
            print("Calculated chemical potential limits: \n")
            print(df)

        return df

    @property
    def chem_limits(self) -> dict:
        """Returns the calculated chemical potential limits"""
        if not hasattr(self, "_chem_limits"):
            self.calculate_chempots()
        return self._chem_limits

    @property
    def intrinsic_chem_limits(self) -> dict:
        """Returns the calculated intrinsic chemical potential limits"""
        if not hasattr(self, "_intrinsic_chem_limits"):
            self.calculate_chempots()
        return self._intrinsic_chem_limits

    @property
    def intrinsic_phase_diagram(self) -> dict:
        """Returns the calculated intrinsic phase diagram"""
        if not hasattr(self, "_intrinsic_phase_diagram"):
            self.calculate_chempots()
        return self._intrinsic_phase_diagram

    def cplap_input(self, dependent_variable=None, filename="input.dat"):
        """For completeness' sake, automatically saves to input.dat for cplap
        Args:
            dependent_variable (str) Pick one of the variables as dependent, the first element is
                chosen from the composition if this isn't set
            filename (str): filename, should end in .dat
        Returns
            None, writes input.dat file
        """
        if not hasattr(self, "chem_limits"):
            self.calculate_chempots(verbose=False)

        with open(filename, "w") as f:
            with contextlib.redirect_stdout(f):
                # get lowest energy bulk phase
                bulk_entries = [
                    sub_dict
                    for sub_dict in self.data
                    if self.bulk_composition.reduced_composition
                    == Composition(sub_dict["formula"]).reduced_composition
                ]
                bulk_entry = min(bulk_entries, key=lambda x: x["formation_energy"])
                print(
                    f"{len(self.bulk_composition.as_dict())}  # number of elements in bulk"
                )
                for k, v in self.bulk_composition.as_dict().items():
                    print(int(v), k, end=" ")
                print(
                    f"{bulk_entry['formation_energy']}  # num_atoms, element, formation_energy "
                    "(bulk)"
                )

                if dependent_variable is not None:
                    print(f"{dependent_variable}  # dependent variable (element)")
                else:
                    print(f"{self.elemental[0]}  # dependent variable (element)")

                # get only the lowest energy entries of compositions in self.data which are on a
                # facet in self._intrinsic_chem_limits
                bordering_phases = {
                    phase
                    for facet in self._chem_limits["facets"]
                    for phase in facet.split("-")
                }
                entries_for_cplap = [
                    entry_dict
                    for entry_dict in self.data
                    if entry_dict["formula"] in bordering_phases
                    and Composition(entry_dict["formula"]).reduced_composition
                    != self.bulk_composition.reduced_composition
                ]
                # cull to only the lowest energy entries of each composition
                culled_cplap_entries = {}
                for entry in entries_for_cplap:
                    reduced_comp = Composition(entry["formula"]).reduced_composition
                    if reduced_comp not in culled_cplap_entries:
                        culled_cplap_entries[reduced_comp] = entry
                    else:
                        if (
                            entry["formation_energy"]
                            < culled_cplap_entries[reduced_comp]["formation_energy"]
                        ):
                            culled_cplap_entries[reduced_comp] = entry

                print(f"{len(culled_cplap_entries)}  # number of bordering phases")
                for i in culled_cplap_entries.values():
                    print(
                        f"{len(Composition(i['formula']).as_dict())}  # number of elements in "
                        "phase:"
                    )
                    for k, v in Composition(i["formula"]).as_dict().items():
                        print(int(v), k, end=" ")
                    print(
                        f"{i['formation_energy']}  # num_atoms, element, formation_energy"
                    )


def combine_extrinsic(first, second, extrinsic_species):
    # TODO: Can we just integrate this to `CompetingPhaseAnalyzer`, so you just pass in a list of
    # extrinsic species and it does the right thing?
    """
    Combines chemical limits for different extrinsic species using chemical limits json file from
    ChemicalPotentialAnalysis. Usage explained in the example jupyter notebook
    Args:
        first (dict): First chemical potential dictionary, it can contain extrinsic species other
        than the set extrinsic species
        second (dict): Second chemical potential dictionary, it must contain the extrinsic species
        extrinsic_species (str): Extrinsic species in the second dictionary
    Returns:
        dict
    """
    keys = ["elemental_refs", "facets", "facets_wrt_el_refs"]
    if not all(key in first for key in keys):
        raise KeyError(
            "the first dictionary doesn't contain the correct keys - it should include "
            "elemental_refs, facets and facets_wrt_el_refs"
        )

    if not all(key in second for key in keys):
        raise KeyError(
            "the second dictionary doesn't contain the correct keys - it should include "
            "elemental_refs, facets and facets_wrt_el_refs"
        )

    if extrinsic_species not in second["elemental_refs"].keys():
        raise ValueError("extrinsic species is not present in the second dictionary")

    cpa1 = copy.deepcopy(first)
    cpa2 = copy.deepcopy(second)
    new_facets = {}
    for (k1, v1), (k2, v2) in zip(
        list(cpa1["facets"].items()), list(cpa2["facets"].items())
    ):
        if k2.rsplit("-", 1)[0] in k1:
            new_key = k1 + "-" + k2.rsplit("-", 1)[1]
        else:
            raise ValueError(
                "The facets aren't matching, make sure you've used the correct dicitonary"
            )

        v1[extrinsic_species] = v2.pop(extrinsic_species)
        new_facets[new_key] = v1

    new_facets_wrt_elt = {}
    for (k1, v1), (k2, v2) in zip(
        list(cpa1["facets_wrt_el_refs"].items()),
        list(cpa2["facets_wrt_el_refs"].items()),
    ):
        if k2.rsplit("-", 1)[0] in k1:
            new_key = k1 + "-" + k2.rsplit("-", 1)[1]
        else:
            raise ValueError(
                "The facets aren't matching, make sure you've used the correct dicitonary"
            )

        v1[extrinsic_species] = v2.pop(extrinsic_species)
        new_facets_wrt_elt[new_key] = v1

    new_elements = copy.deepcopy(cpa1["elemental_refs"])
    new_elements[extrinsic_species] = copy.deepcopy(cpa2["elemental_refs"])[
        extrinsic_species
    ]

    new_dict = {}
    new_dict = {
        "elemental_refs": new_elements,
        "facets": new_facets,
        "facets_wrt_el_refs": new_facets_wrt_elt,
    }

    return new_dict
