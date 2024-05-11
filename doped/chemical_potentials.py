"""
Functions for setting up and parsing competing phase calculations in order to
determine and analyse the elemental chemical potentials for defect formation
energies.
"""

import contextlib
import copy
import os
import warnings
from pathlib import Path, PurePath

import numpy as np
import pandas as pd
from monty.serialization import loadfn
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import SETTINGS, Composition, Element, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import UnconvergedVASPWarning
from tqdm import tqdm

from doped import _ignore_pmg_warnings
from doped.utils.parsing import _get_output_files_and_check_if_multiple, get_vasprun
from doped.vasp import MODULE_DIR, DopedDictSet, default_HSE_set, default_relax_set

pbesol_convrg_set = loadfn(os.path.join(MODULE_DIR, "VASP_sets/PBEsol_ConvergenceSet.yaml"))  # just INCAR

# globally ignore:
_ignore_pmg_warnings()
warnings.filterwarnings(
    "ignore", message="You are using the legacy MPRester"
)  # currently rely on this so shouldn't show warning, `message` only needs to match start of message


# TODO: Need to recheck all functionality from old `_chemical_potentials.py` is now present here.
# TODO: Add chemical potential diagram plotting functionality that we had before
#  with `plot_cplap_ternary` -- using ``ChemicalPotentialGrid`` from Alex PR; code from
#  pymatgen/analysis/defects/plotting/phases.py may be useful


def make_molecule_in_a_box(element: str):
    """
    Generate an X2 'molecule-in-a-box' structure for the input element X, (i.e.
    a 30 Å cuboid supercell with a single X2 molecule in the centre).

    This is the natural state of several elemental competing phases, such
    as O2, N2, H2, F2 and Cl2. Initial bond lengths are set to the experimental
    bond lengths of these gaseous molecules.

    Args:
        element (str):
            Element symbol of the molecule to generate.

    Returns:
        Structure, formula and total magnetization:

        structure (Structure):
            ``pymatgen`` ``Structure`` object of the molecule in a box.
        formula (str):
            Chemical formula of the molecule in a box.
        total_magnetization (int):
            Total magnetization of the molecule in a box
            (0 for all X2 except O2 which has a triplet ground state (S = 1)).
    """
    lattice = [[30.01, 0, 0], [0, 30.00, 0], [0, 0, 29.99]]
    all_structures = {
        "O2": {
            "structure": Structure(
                lattice=lattice,
                species=["O", "O"],
                coords=[[15, 15, 15], [15, 15, 16.21]],
                coords_are_cartesian=True,
            ),
            "formula": "O2",
            "total_magnetization": 2,
        },
        "N2": {
            "structure": Structure(
                lattice=lattice,
                species=["N", "N"],
                coords=[[15, 15, 15], [15, 15, 16.10]],
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
                coords=[[15, 15, 15], [15, 15, 16.42]],
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

    if element not in all_structures:
        raise ValueError(
            f"Element {element} is not currently supported for molecule-in-a-box structure generation."
        )

    structure = all_structures[element]["structure"]
    formula = all_structures[element]["formula"]
    total_magnetization = all_structures[element]["total_magnetization"]

    return structure, formula, total_magnetization


def _make_molecular_entry(computed_entry):
    """
    Generate a new ComputedStructureEntry for a molecule in a box, for the
    input elemental ComputedEntry.
    """
    assert len(computed_entry.composition.elements) == 1  # Elemental!
    struct, formula, total_magnetization = make_molecule_in_a_box(computed_entry.data["pretty_formula"])
    molecular_entry = ComputedStructureEntry(
        structure=struct,
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


def _calculate_formation_energies(data: list, elemental: dict):
    """
    Calculate formation energies for a list of dictionaries, using the input
    elemental reference energies.

    Args:
        data (list): List of dictionaries containing the energy data of the
            phases to calculate formation energies for.
        elemental (dict): Dictionary of elemental reference energies.

    Returns:
        pd.DataFrame: DataFrame formation energies of the input phases.
    """
    for d in data:
        for el in elemental:
            d[el] = Composition(d["Formula"]).as_dict().get(el, 0)

    formation_energy_df = pd.DataFrame(data)
    formation_energy_df["num_atoms_in_fu"] = formation_energy_df["Formula"].apply(
        lambda x: Composition(x).num_atoms
    )
    formation_energy_df["num_species"] = formation_energy_df["Formula"].apply(
        lambda x: len(Composition(x).as_dict())
    )

    # get energy per fu then subtract elemental energies later, to get formation energies
    if "DFT Energy (eV/fu)" in formation_energy_df.columns:
        formation_energy_df["formation_energy_calc"] = formation_energy_df["DFT Energy (eV/fu)"]
        if "DFT Energy (eV/atom)" not in formation_energy_df.columns:
            formation_energy_df["DFT Energy (eV/atom)"] = formation_energy_df["DFT Energy (eV/fu)"] / (
                formation_energy_df["num_atoms_in_fu"]
            )

    elif "DFT Energy (eV/atom)" in formation_energy_df.columns:
        formation_energy_df["formation_energy_calc"] = (
            formation_energy_df["DFT Energy (eV/atom)"] * formation_energy_df["num_atoms_in_fu"]
        )
        formation_energy_df["DFT Energy (eV/fu)"] = formation_energy_df["DFT Energy (eV/atom)"] * (
            formation_energy_df["num_atoms_in_fu"]
        )

    else:
        raise ValueError(
            "No energy data (DFT Energy (eV/atom) or per Formula Unit (eV/fu)) found in input "
            "data to calculate formation energies!"
        )

    for k, v in elemental.items():
        formation_energy_df["formation_energy_calc"] -= formation_energy_df[k] * v

    formation_energy_df["Formation Energy (eV/fu)"] = formation_energy_df["formation_energy_calc"]
    formation_energy_df["Formation Energy (eV/atom)"] = (
        formation_energy_df["formation_energy_calc"] / formation_energy_df["num_atoms_in_fu"]
    )
    formation_energy_df = formation_energy_df.drop(columns=["formation_energy_calc"])

    # sort by num_species, then alphabetically, then by num_atoms_in_fu, then by formation_energy
    formation_energy_df = formation_energy_df.sort_values(
        by=["num_species", "Formula", "num_atoms_in_fu", "Formation Energy (eV/fu)"],
    )
    # drop num_atoms_in_fu and num_species
    return formation_energy_df.drop(columns=["num_atoms_in_fu", "num_species"])


def _renormalise_entry(entry, renormalisation_energy_per_atom):
    """
    Regenerate the input entry with an energy per atom decreased by
    renormalisation_energy_per_atom.
    """
    renormalised_entry_dict = entry.as_dict().copy()
    renormalised_entry_dict["energy"] = entry.energy - renormalisation_energy_per_atom * sum(
        entry.composition.values()
    )  # entry.energy includes MP corrections as desired

    return PDEntry.from_dict(renormalised_entry_dict)


def get_chempots_from_phase_diagram(bulk_ce, phase_diagram):
    """
    Get the chemical potential limits for the bulk computed entry in the
    supplied phase diagram.

    Args:
        bulk_ce: Pymatgen ComputedStructureEntry object for bulk entry / supercell
        phase_diagram: Pymatgen PhaseDiagram object for the system of interest
    """
    bulk_composition = bulk_ce.composition
    redcomp = bulk_composition.reduced_composition
    # append bulk_ce to phase diagram, if not present
    entries = phase_diagram.all_entries
    if not any(
        (ent.composition == bulk_ce.composition and ent.energy == bulk_ce.energy) for ent in entries
    ):
        entries.append(
            PDEntry(
                bulk_ce.composition,
                bulk_ce.energy,
                attribute="Bulk Material",
            )
        )
        phase_diagram = PhaseDiagram(entries)

    return phase_diagram.get_all_chempots(redcomp)


class CompetingPhases:
    # TODO: Need to add functionality to deal with cases where the bulk composition is not listed
    # on the MP - warn user (i.e. check your stuff) and generate the competing phases according to
    # composition position within phase diagram. (i.e. downshift it to the convex hull, print warning
    # and generate from there)
    # E.g. from pycdt chemical_potentials:
    # #                 "However, no stable entry with this composition exists "
    # #                 "in the MP database!\nPlease consider submitting the "
    # #                 "POSCAR to the MP xtaltoolkit, so future users will "
    # #                 "know about this structure:"
    # #                 " https://materialsproject.org/#apps/xtaltoolkit\n" - see
    # analyze_GGA_chempots code for example.
    # e.g. in pmg-analysis-defects (for parsing, like our approach): stable_entry = ComputedEntry(
    #         entry.composition,
    #         pd.get_hull_energy(entry.composition) - threshold,
    #     )
    # Na2FePO4F a good test case for this, 0.17 eV/atom above the MP Hull

    def __init__(self, composition, e_above_hull=0.1, api_key=None, full_phase_diagram=False):
        """
        Class to generate the VASP input files for competing phases on the
        phase diagram for the host material, which determine the chemical
        potential limits for that compound.

        For this, the Materials Project (MP) database is queried using the
        ``MPRester`` API, and any calculated compounds which _could_ border
        the host material within an error tolerance for the semi-local DFT
        database energies (``e_above_hull``, 0.1 eV/atom by default) are
        generated, along with the elemental reference phases.
        Diatomic gaseous molecules are generated as molecules-in-a-box as
        appropriate (e.g. for O2, F2, H2 etc).

        Args:
            composition (str, ``Composition``):
                Composition of the host material (e.g. ``'LiFePO4'``, or
                ``Composition('LiFePO4')``, or
                ``Composition({"Li":1, "Fe":1, "P":1, "O":4})``).
            e_above_hull (float):
                Maximum energy above hull (in eV/atom) of Materials Project
                entries to be considered as competing phases. This is an
                uncertainty range for the MP-calculated formation energies,
                which may not be accurate due to functional choice (GGA vs
                hybrid DFT / GGA+U / RPA etc.), lack of vdW corrections etc.
                Any phases that (would) border the host material on the phase
                diagram, if their relative energy was downshifted by
                ``e_above_hull``, are included.
                (Default is 0.1 eV/atom).
            api_key (str):
                Materials Project (MP) API key, needed to access the MP
                database for competing phase generation. If not supplied, will
                attempt to read from environment variable ``PMG_MAPI_KEY`` (in
                ``~/.pmgrc.yaml`` or ``~/.config/.pmgrc.yaml``) - see the ``doped``
                Installation docs page:
                https://doped.readthedocs.io/en/latest/Installation.html
            full_phase_diagram (bool):
                If ``True``, include all phases on the MP phase diagram (with energy
                above hull < ``e_above_hull`` eV/atom) for the chemical system of
                the input composition (not recommended). If ``False``, only includes
                phases that would border the host material on the phase diagram (and
                thus set the chemical potential limits), if their relative energy was
                downshifted by ``e_above_hull`` eV/atom.
                (Default is False).
        """
        self.api_key = api_key or SETTINGS.get("PMG_MAPI_KEY")

        # create list of entries
        self._molecules_in_a_box = ["H2", "O2", "N2", "F2", "Cl2"]

        # TODO: Should hard code S (solid + S8), P, Te and Se in here too. Common anions with a
        #  lot of unnecessary polymorphs on MP. Should at least scan over elemental phases and hard code
        #  any particularly bad cases. E.g. P_EaH=0 is red phosphorus (HSE06 groundstate), P_EaH=0.037
        #  is black phosphorus (thermo stable at RT), so only need to generate these. Same for all
        #  alkali and alkaline earth metals (ask the battery boyos), TiO2, SnO2, WO3 (particularly bad
        #  cases).
        # Can have a data file with a list of known, common cases?
        # With Materials Project querying, can check if the structure has a database ID (i.e. is
        # experimentally observed) with icsd_id(s) / theoretical (same thing). Could have 'lean' option
        # which only outputs phases which are either on the MP-predicted hull or have an ICSD ID?
        # Would want to test this to see if it is sufficient in most cases, then can recommend its use
        # with a caution... From a quick test, this does cut down a good chunk of unnecessary phases,
        # but still not all as often there are several ICSD phases for e.g. metals with a load of known
        # polymorphs (at different temperatures/pressures).
        # Strategies for dealing with these cases where MP has many low energy polymorphs in general?
        # Will mention some good practice in the docs anyway. -> Have an in-built warning when many
        # entries for the same composition, warn the user (that if the groundstate phase at low/room
        # temp is well-known, then likely best to prune to that) and direct to relevant section on the
        # docs discussing this
        # - Could have two optional EaH tolerances, a tight one (0.02 eV/atom?) that applies to all,
        # and a looser one (0.1 eV/atom?) that applies to phases with ICSD IDs?

        # all data collected from materials project
        self.data = [  # can see available fields with MPRester.*.available_fields on new API
            "pretty_formula",
            "e_above_hull",
            "band_gap",
            "nsites",
            "volume",
            "icsd_id",
            "icsd_ids",  # some entries have icsd_id and some have icsd_ids
            "theoretical",
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
        if self.api_key is None:  # no API key supplied or set in ``.pmgrc.yaml``
            raise ValueError(
                "No API key (``api_key`` parameter or 'PMG_MAPI_KEY' in the ``~/.pmgrc.yaml`` or "
                "``~/.config/.pmgrc.yaml file) was supplied. This is required for automatic competing "
                "phase generation in doped, as detailed on the installation instructions:\n"
                "https://doped.readthedocs.io/en/latest/Installation.html#setup-potcars-and-materials"
                "-project-api"
            )
        if len(self.api_key) == 32:
            raise ValueError(
                f"The supplied API key (``api_key`` or 'PMG_MAPI_KEY' in your ``~/.pmgrc.yaml`` or "
                f"``~/.config/.pmgrc.yaml file; {self.api_key}) corresponds to the new Materials Project "
                f"(MP) API, which is not supported by doped. Please use the legacy MP API as detailed on "
                f"the doped installation instructions:\n"
                f"https://doped.readthedocs.io/en/latest/Installation.html#setup-potcars-and-materials"
                f"-project-api"
            )
        if 15 <= len(self.api_key) <= 20:
            self.eah = "e_above_hull"
        else:
            raise ValueError(
                f"The supplied API key (``api_key`` or 'PMG_MAPI_KEY' in your ``~/.pmgrc.yaml`` or "
                f"``~/.config/.pmgrc.yaml file; {self.api_key}) is not a valid legacy Materials Project "
                f"API key, which is required by doped. See the doped installation instructions for "
                f"details:\n"
                "https://doped.readthedocs.io/en/latest/Installation.html#setup-potcars-and-materials"
                "-project-api"
            )

        # use with MPRester() as mpr: if self.api_key is None, else use with MPRester(self.api_key)
        with contextlib.ExitStack() as stack:
            if self.api_key is None:
                mpr = stack.enter_context(MPRester())
            else:
                mpr = stack.enter_context(MPRester(self.api_key))

            # get all entries in the chemical system
            self.MP_full_pd_entries = mpr.get_entries_in_chemsys(
                list(self.bulk_comp.as_dict().keys()),
                inc_structure="initial",
                property_data=self.data,
            )

        self.MP_full_pd_entries.sort(key=lambda x: x.data["e_above_hull"])  # sort by e_above_hull
        self.MP_full_pd = PhaseDiagram(self.MP_full_pd_entries)

        formatted_pd_entries = []
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
                    and ent.data["pretty_formula"] == molecular_entry.data["pretty_formula"]
                    for ent in formatted_pd_entries
                ):  # first entry only
                    formatted_pd_entries.append(molecular_entry)
                    self.MP_full_pd_entries.append(molecular_entry)

            elif entry.data["pretty_formula"] not in self._molecules_in_a_box:
                entry.data["molecule"] = False
                formatted_pd_entries.append(entry)

        formatted_pd_entries.sort(key=lambda x: x.energy_per_atom)  # sort by energy per atom
        temp_phase_diagram = PhaseDiagram(formatted_pd_entries)
        for entry in formatted_pd_entries:
            # reparse energy above hull, to avoid mislabelling issues noted in Materials Project database
            entry.data["e_above_hull"] = temp_phase_diagram.get_e_above_hull(entry)
        pd_entries = [
            entry for entry in temp_phase_diagram.all_entries if entry.data["e_above_hull"] <= e_above_hull
        ]
        phase_diagram = PhaseDiagram(pd_entries)
        bulk_entries = [  # TODO: Currently breaks if bulk composition not on MP, need to fix!
            entry
            for entry in pd_entries
            if entry.composition.reduced_composition == self.bulk_comp.reduced_composition
        ]
        bulk_ce = bulk_entries[0]  # lowest energy entry for bulk composition (after sorting)
        self.MP_bulk_ce = bulk_ce

        if not full_phase_diagram:  # default
            # cull to only include any phases that would border the host material on the phase
            # diagram, if their relative energy was downshifted by ``e_above_hull``:
            MP_gga_chempots = get_chempots_from_phase_diagram(bulk_ce, phase_diagram)

            MP_bordering_phases = {phase for limit in MP_gga_chempots for phase in limit.split("-")}
            self.entries = [
                entry for entry in pd_entries if entry.name in MP_bordering_phases or entry.is_element
            ]

            # add any phases that would border the host material on the phase diagram, if their
            # relative energy was downshifted by ``e_above_hull``:
            for entry in pd_entries:
                if entry.name not in MP_bordering_phases and not entry.is_element:
                    # decrease entry energy per atom by ``e_above_hull`` eV/atom
                    renormalised_entry = _renormalise_entry(entry, e_above_hull)
                    new_phase_diagram = PhaseDiagram([*phase_diagram.entries, renormalised_entry])
                    new_MP_gga_chempots = get_chempots_from_phase_diagram(bulk_ce, new_phase_diagram)

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

    # TODO: Return dict of DictSet objects for this and vasp_std_setup() functions, as well as
    #  write_files option, for ready integration with high-throughput workflows
    def convergence_setup(
        self,
        kpoints_metals=(40, 1000, 5),
        kpoints_nonmetals=(5, 120, 5),
        user_potcar_functional="PBE",
        user_potcar_settings=None,
        user_incar_settings=None,
        **kwargs,
    ):
        """
        Generates VASP input files for k-points convergence testing for
        competing phases, using PBEsol (GGA) DFT by default. Automatically sets
        the ``ISMEAR`` ``INCAR`` tag to 2 (if metallic) or 0 if not. Recommended to
        use with https://github.com/kavanase/vaspup2.0.

        Args:
            kpoints_metals (tuple):
                Kpoint density per inverse volume (Å^-3) to be tested in
                (min, max, step) format for metals
            kpoints_nonmetals (tuple):
                Kpoint density per inverse volume (Å^-3) to be tested in
                (min, max, step) format for nonmetals
            user_potcar_functional (str):
                POTCAR functional to use. Default is "PBE" and if this fails,
                tries "PBE_52", then "PBE_54".
            user_potcar_settings (dict):
                Override the default POTCARs, e.g. {"Li": "Li_sv"}. See
                ``doped/VASP_sets/PotcarSet.yaml`` for the default ``POTCAR`` set.
            user_incar_settings (dict):
                Override the default INCAR settings e.g. {"EDIFF": 1e-5, "LDAU": False,
                "ALGO": "All"}. Note that any non-numerical or non-True/False flags
                need to be input as strings with quotation marks. See
                ``doped/VASP_sets/PBEsol_ConvergenceSet.yaml`` for the default settings.
            **kwargs: Additional kwargs to pass to ``DictSet.write_input()``
        """
        # by default uses pbesol, but easy to switch to pbe or pbe+u using user_incar_settings

        # kpoints should be set as (min, max, step)
        min_nm, max_nm, step_nm = kpoints_nonmetals
        min_m, max_m, step_m = kpoints_metals

        base_user_incar_settings = copy.deepcopy(pbesol_convrg_set["INCAR"])
        base_user_incar_settings.update(user_incar_settings or {})  # user_incar_settings override defaults

        # separate metals and non-metals
        self.nonmetals = []
        self.metals = []
        for e in self.entries:
            if e.data["molecule"]:
                print(f"{e.name} is a molecule in a box, does not need convergence testing")

            elif e.data["band_gap"] > 0:
                self.nonmetals.append(e)
            else:
                self.metals.append(e)

        for e in self.nonmetals:
            uis = copy.deepcopy(base_user_incar_settings)  # don't overwrite base_user_incar_settings
            self._set_spin_polarisation(uis, user_incar_settings or {}, e)

            dict_set = DopedDictSet(  # use ``doped`` DopedDictSet for quicker IO functions
                structure=e.structure,
                user_incar_settings=uis,
                user_kpoints_settings={"reciprocal_density": min_nm},
                user_potcar_settings=user_potcar_settings or {},
                user_potcar_functional=user_potcar_functional,
                force_gamma=True,
            )

            for kpoint in range(min_nm, max_nm, step_nm):
                dict_set.user_kpoints_settings = {"reciprocal_density": kpoint}
                kname = (
                    "k"
                    + ("_" * (dict_set.kpoints.kpts[0][0] // 10))
                    + ",".join(str(k) for k in dict_set.kpoints.kpts[0])
                )
                fname = f"competing_phases/{self._competing_phase_name(e)}/kpoint_converge/{kname}"
                # TODO: competing_phases folder name should be an optional parameter, and rename default
                #  to something that isn't so ugly? CompetingPhases?
                # TODO: Naming should be done in __init__ to ensure consistency and efficiency. Watch
                #  out for cases where rounding can give same name (e.g. Te!) - should use
                #  {formula}_MP_{mpid}_EaH_{round(e_above_hull,4)} as naming convention, to prevent any
                #  rare cases of overwriting
                dict_set.write_input(fname, **kwargs)

        for e in self.metals:
            uis = copy.deepcopy(base_user_incar_settings)  # don't overwrite base_user_incar_settings
            self._set_spin_polarisation(uis, user_incar_settings or {}, e)
            self._set_default_metal_smearing(uis, user_incar_settings or {})

            dict_set = DopedDictSet(  # use ``doped`` DopedDictSet for quicker IO functions
                structure=e.structure,
                user_kpoints_settings={"reciprocal_density": min_m},
                user_incar_settings=uis,
                user_potcar_settings=user_potcar_settings or {},
                user_potcar_functional=user_potcar_functional,
                force_gamma=True,
            )

            for kpoint in range(min_m, max_m, step_m):
                dict_set.user_kpoints_settings = {"reciprocal_density": kpoint}
                kname = (
                    "k"
                    + ("_" * (dict_set.kpoints.kpts[0][0] // 10))
                    + ",".join(str(k) for k in dict_set.kpoints.kpts[0])
                )
                fname = f"competing_phases/{self._competing_phase_name(e)}/kpoint_converge/{kname}"
                dict_set.write_input(fname, **kwargs)

    def _competing_phase_name(self, entry):
        rounded_eah = round(entry.data["e_above_hull"], 4)
        if np.isclose(rounded_eah, 0):
            return f"{entry.name}_EaH_0"
        return f"{entry.name}_EaH_{rounded_eah}"

    # TODO: Add vasp_ncl_setup()
    def vasp_std_setup(
        self,
        kpoints_metals=200,
        kpoints_nonmetals=64,  # MPRelaxSet default
        user_potcar_functional="PBE",
        user_potcar_settings=None,
        user_incar_settings=None,
        **kwargs,
    ):
        """
        Generates VASP input files for ``vasp_std`` relaxations of the
        competing phases, using HSE06 (hybrid DFT) DFT by default.
        Automatically sets the ``ISMEAR`` ``INCAR`` tag to 2 (if metallic) or 0
        if not. Note that any changes to the default ``INCAR``/``POTCAR``
        settings should be consistent with those used for the defect supercell
        calculations.

        Args:
            kpoints_metals (int):
                Kpoint density per inverse volume (Å^-3) for metals.
                Default is 200.
            kpoints_nonmetals (int):
                Kpoint density per inverse volume (Å^-3) for nonmetals
                (default is 64, the default for ``MPRelaxSet``).
            user_potcar_functional (str):
                POTCAR functional to use. Default is "PBE" and if this fails,
                tries "PBE_52", then "PBE_54".
            user_potcar_settings (dict):
                Override the default POTCARs, e.g. {"Li": "Li_sv"}. See
                ``doped/VASP_sets/PotcarSet.yaml`` for the default ``POTCAR`` set.
            user_incar_settings (dict):
                Override the default INCAR settings e.g. {"EDIFF": 1e-5, "LDAU": False,
                "ALGO": "All"}. Note that any non-numerical or non-True/False flags
                need to be input as strings with quotation marks. See
                ``doped/VASP_sets/RelaxSet.yaml`` and ``HSESet.yaml`` for the default settings.
            **kwargs: Additional kwargs to pass to ``DictSet.write_input()``
        """
        base_incar_settings = copy.deepcopy(default_relax_set["INCAR"])

        lhfcalc = (
            True if user_incar_settings is None else user_incar_settings.get("LHFCALC", True)
        )  # True (hybrid) by default for vasp_std relaxations
        if lhfcalc or (isinstance(lhfcalc, str) and lhfcalc.lower().startswith("t")):
            base_incar_settings.update(default_HSE_set["INCAR"])

        base_incar_settings.update(user_incar_settings or {})  # user_incar_settings override defaults

        # separate metals, non-metals and molecules
        self.nonmetals = []
        self.metals = []
        self.molecules = []
        for e in self.entries:
            if e.data["molecule"]:
                self.molecules.append(e)
            elif e.data["band_gap"] > 0:
                self.nonmetals.append(e)
            else:
                self.metals.append(e)

        for e in self.nonmetals:
            uis = copy.deepcopy(base_incar_settings or {})
            self._set_spin_polarisation(uis, user_incar_settings or {}, e)

            dict_set = DopedDictSet(  # use ``doped`` DopedDictSet for quicker IO functions
                structure=e.structure,
                user_incar_settings=uis,
                user_kpoints_settings={"reciprocal_density": kpoints_nonmetals},
                user_potcar_settings=user_potcar_settings or {},
                user_potcar_functional=user_potcar_functional,
                force_gamma=True,
            )

            fname = f"competing_phases/{self._competing_phase_name(e)}/vasp_std"
            dict_set.write_input(fname, **kwargs)

        for e in self.metals:
            uis = copy.deepcopy(base_incar_settings or {})
            self._set_spin_polarisation(uis, user_incar_settings or {}, e)
            self._set_default_metal_smearing(uis, user_incar_settings or {})

            dict_set = DopedDictSet(  # use ``doped`` DopedDictSet for quicker IO functions
                structure=e.structure,
                user_incar_settings=uis,
                user_kpoints_settings={"reciprocal_density": kpoints_metals},
                user_potcar_settings=user_potcar_settings or {},
                user_potcar_functional=user_potcar_functional,
                force_gamma=True,
            )

            fname = f"competing_phases/{self._competing_phase_name(e)}/vasp_std"
            dict_set.write_input(fname, **kwargs)

        for e in self.molecules:  # gamma-only for molecules
            uis = copy.deepcopy(base_incar_settings or {})
            uis["ISIF"] = 2  # can't change the volume
            uis["KPAR"] = 1  # can't use k-point parallelization, gamma only
            self._set_spin_polarisation(uis, user_incar_settings or {}, e)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="KPOINTS are Γ-only")  # Γ only KPAR warning
                dict_set = DopedDictSet(  # use ``doped`` DopedDictSet for quicker IO functions
                    structure=e.structure,  # molecule in a box structure
                    user_incar_settings=uis,
                    user_kpoints_settings=Kpoints().from_dict(
                        {
                            "comment": "Gamma-only kpoints for molecule-in-a-box",
                            "generation_style": "Gamma",
                        }
                    ),
                    user_potcar_settings=user_potcar_settings or {},
                    user_potcar_functional=user_potcar_functional,
                    force_gamma=True,
                )
                fname = f"competing_phases/{self._competing_phase_name(e)}/vasp_std"
                dict_set.write_input(fname, **kwargs)

    def _set_spin_polarisation(self, incar_settings, user_incar_settings, entry):
        """
        If the entry has a non-zero total magnetisation (greater than the
        default tolerance of 0.1), set ``ISPIN`` to 2 (allowing spin
        polarisation) and ``NUPDOWN`` equal to the integer-rounded total
        magnetisation.

        See
        https://doped.readthedocs.io/en/latest/Tips.html#spin-polarisation
        """
        if entry.data["total_magnetization"] > 0.1:  # account for magnetic moment
            incar_settings["ISPIN"] = user_incar_settings.get("ISPIN", 2)
            if "NUPDOWN" not in incar_settings and int(entry.data["total_magnetization"]) > 0:
                incar_settings["NUPDOWN"] = int(entry.data["total_magnetization"])

    def _set_default_metal_smearing(self, incar_settings, user_incar_settings):
        """
        Set the smearing parameters to the ``doped`` defaults for metallic
        phases (i.e. ``ISMEAR`` = 2 (Methfessel-Paxton) and ``SIGMA`` = 0.2
        eV).
        """
        incar_settings["ISMEAR"] = user_incar_settings.get("ISMEAR", 2)
        incar_settings["SIGMA"] = user_incar_settings.get("SIGMA", 0.2)


# TODO: Add full_sub_approach option
# TODO: Add warnings for full_sub_approach=True, especially if done with multiple
#  extrinsic species.
class ExtrinsicCompetingPhases(CompetingPhases):
    """
    This class generates the competing phases that need to be calculated to
    obtain the chemical potential limits when doping with extrinsic species /
    impurities.

    Ensures that only the necessary additional competing phases are generated.
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
                relative energy was downshifted by ``e_above_hull``, are included.
                Default is 0.1 eV/atom.
            full_sub_approach (bool): Generate competing phases by considering the full phase
                diagram, including chemical potential limits with multiple extrinsic phases.
                Only recommended when looking at high (non-dilute) doping concentrations.
                Default = False. Described in further detail below.
            codoping (bool): Whether to consider extrinsic competing phases containing multiple
                extrinsic species. Only relevant to high (non-dilute) co-doping concentrations.
                If set to True, then ``full_sub_approach`` is also set to True.
                Default = False.
            api_key (str): Materials Project (MP) API key, needed to access the MP database for
                competing phase generation. If not supplied, will attempt to read from
                environment variable ``PMG_MAPI_KEY`` (in ``~/.pmgrc.yaml``) - see the ``doped``
                Installation docs page: https://doped.readthedocs.io/en/latest/Installation.html
                This should correspond to the legacy MP API; from
                https://legacy.materialsproject.org/open.

        This code uses the Materials Project (MP) phase diagram data along with the
        ``e_above_hull`` error range to generate potential competing phases.

        NOTE on 'full_sub_approach':
            The default approach for substitutional elements (``full_sub_approach = False``) is to
            only consider chemical potential limits with a maximum of 1 extrinsic phase
            (composition with extrinsic species present). This is a valid approximation for the
            case of dilute dopant/impurity concentrations. For high (non-dilute) concentrations
            of extrinsic species, use ``full_sub_approach = True``.
        """
        # the competing phases & entries of the OG system
        super().__init__(composition, e_above_hull, api_key)
        self.intrinsic_entries = copy.deepcopy(self.entries)
        self.entries = []
        self.intrinsic_species = [s.symbol for s in self.bulk_comp.reduced_composition.elements]
        self.MP_intrinsic_full_pd_entries = self.MP_full_pd_entries  # includes molecules-in-boxes

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

        if full_sub_approach:  # can be time-consuming if several extrinsic_species supplied
            if codoping:
                # TODO: `full_sub_approach` shouldn't necessarily mean `full_phase_diagram =
                #  True` right? As in can be non-full-phase-diagram intrinsic + extrinsic
                #  entries, including limits with multiple extrinsic entries but still not the
                #  full phase diagram? - To be updated!
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

                with contextlib.ExitStack() as stack:
                    if self.api_key is None:
                        mpr = stack.enter_context(MPRester())
                    else:
                        mpr = stack.enter_context(MPRester(self.api_key))

                    # get all entries in the chemical system
                    self.MP_full_pd_entries = mpr.get_entries_in_chemsys(
                        self.intrinsic_species + self.extrinsic_species,
                        inc_structure="initial",
                        property_data=self.data,
                    )
                self.MP_full_pd_entries = [
                    e for e in self.MP_full_pd_entries if e.data["e_above_hull"] <= e_above_hull
                ]

                # sort by e_above_hull:
                self.MP_full_pd_entries.sort(key=lambda x: x.data["e_above_hull"])

                for entry in self.MP_full_pd_entries.copy():
                    if any(sub_el in entry.composition for sub_el in self.extrinsic_species):
                        if (
                            entry.data["pretty_formula"] in self._molecules_in_a_box
                            and entry.data["e_above_hull"] == 0
                        ):  # only first matching entry
                            molecular_entry = _make_molecular_entry(entry)
                            if not any(
                                entry.data["molecule"]
                                and entry.data["pretty_formula"] == molecular_entry.data["pretty_formula"]
                                for entry in self.entries
                            ):  # first entry only
                                self.MP_full_pd_entries.append(molecular_entry)
                                self.entries.append(molecular_entry)
                        elif entry.data["pretty_formula"] not in self._molecules_in_a_box:
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
                for sub_el in self.extrinsic_species:
                    extrinsic_entries = []
                    with contextlib.ExitStack() as stack:
                        if self.api_key is None:
                            mpr = stack.enter_context(MPRester())
                        else:
                            mpr = stack.enter_context(MPRester(self.api_key))

                        MP_full_pd_entries = mpr.get_entries_in_chemsys(
                            [*self.intrinsic_species, sub_el],
                            inc_structure="initial",
                            property_data=self.data,
                        )
                    MP_full_pd_entries = [
                        e for e in MP_full_pd_entries if e.data["e_above_hull"] <= e_above_hull
                    ]
                    # sort by e_above_hull:
                    MP_full_pd_entries.sort(key=lambda x: x.data["e_above_hull"])

                    for entry in MP_full_pd_entries.copy():
                        if entry not in self.MP_full_pd_entries:
                            self.MP_full_pd_entries.append(entry)
                        if sub_el in entry.composition:
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

                            elif entry.data["pretty_formula"] not in self._molecules_in_a_box:
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

            for sub_el in self.extrinsic_species:
                extrinsic_pd_entries = []
                with contextlib.ExitStack() as stack:
                    if self.api_key is None:
                        mpr = stack.enter_context(MPRester())
                    else:
                        mpr = stack.enter_context(MPRester(self.api_key))

                    MP_full_pd_entries = mpr.get_entries_in_chemsys(
                        [*self.intrinsic_species, sub_el],
                        inc_structure="initial",
                        property_data=self.data,
                    )
                self.MP_full_pd_entries = [
                    e for e in MP_full_pd_entries if e.data["e_above_hull"] <= e_above_hull
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
                            and entry.data["pretty_formula"] == molecular_entry.data["pretty_formula"]
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
                        f"system: {[*self.intrinsic_species, sub_el]}"
                    )

                extrinsic_phase_diagram = PhaseDiagram(extrinsic_pd_entries)
                MP_extrinsic_gga_chempots = get_chempots_from_phase_diagram(
                    self.MP_bulk_ce, extrinsic_phase_diagram
                )
                MP_extrinsic_bordering_phases = []

                for limit in MP_extrinsic_gga_chempots:
                    # if the number of intrinsic competing phases for this limit is equal to the
                    # number of species in the bulk composition, then include the extrinsic phase(s)
                    # for this limit (full_sub_approach = False approach)
                    MP_intrinsic_bordering_phases = {
                        phase for phase in limit.split("-") if sub_el not in phase
                    }
                    if len(MP_intrinsic_bordering_phases) == len(self.intrinsic_species):
                        # add to list of extrinsic bordering phases, if not already present:
                        MP_extrinsic_bordering_phases.extend(
                            [
                                phase
                                for phase in limit.split("-")
                                if sub_el in phase and phase not in MP_extrinsic_bordering_phases
                            ]
                        )

                # add any phases that would border the host material on the phase diagram,
                # if their relative energy was downshifted by `e_above_hull`:
                for entry in MP_full_pd_entries:
                    if (
                        entry.name not in MP_extrinsic_bordering_phases
                        and not entry.is_element
                        and sub_el in entry.composition
                    ):
                        # decrease entry energy per atom by `e_above_hull` eV/atom
                        renormalised_entry = _renormalise_entry(entry, e_above_hull)
                        new_extrinsic_phase_diagram = PhaseDiagram(
                            [*extrinsic_phase_diagram.entries, renormalised_entry]
                        )
                        new_MP_extrinsic_gga_chempots = get_chempots_from_phase_diagram(
                            self.MP_bulk_ce, new_extrinsic_phase_diagram
                        )

                        if new_MP_extrinsic_gga_chempots != MP_extrinsic_gga_chempots:
                            # new bordering phase, check if not an over-dependent limit:

                            for limit in new_MP_extrinsic_gga_chempots:
                                if limit not in MP_extrinsic_gga_chempots:
                                    # new limit, check if not an over-dependent limit:
                                    MP_intrinsic_bordering_phases = {
                                        phase for phase in limit.split("-") if sub_el not in phase
                                    }
                                    if len(MP_intrinsic_bordering_phases) == len(self.intrinsic_species):
                                        MP_extrinsic_bordering_phases.extend(
                                            [
                                                phase
                                                for phase in limit.split("-")
                                                if sub_el in phase
                                                and phase not in MP_extrinsic_bordering_phases
                                            ]
                                        )

                extrinsic_entries = [
                    entry
                    for entry in extrinsic_pd_entries
                    if entry.name in MP_extrinsic_bordering_phases
                    or (entry.is_element and sub_el in entry.name)
                ]

                # check that extrinsic competing phases list is not empty (can happen with
                # 'over-dependent' limits); if so then set full_sub_approach = True and re-run
                # the extrinsic phase addition process
                if not extrinsic_entries:
                    warnings.warn(
                        "Determined chemical potentials to be over dependent on an "
                        "extrinsic species. This means we need to revert to "
                        "`full_sub_approach = True`  running now."
                    )
                    full_sub_approach = True
                    extrinsic_entries = [
                        entry for entry in self.MP_full_pd_entries if sub_el in entry.composition
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
    Post-processing competing phases data to calculate chemical potentials.
    """

    # TODO: Allow parsing using pymatgen ComputedEntries as well, to aid interoperability with
    #  high-througput architectures like AiiDA or atomate2. See:
    #  https://github.com/SMTG-Bham/doped/commit/b4eb9a5083a0a2c9596be5ccc57d060e1fcec530
    def __init__(self, system, extrinsic_species=None):
        """
        Args:
            system (str): The  'reduced formula' of the bulk composition
            extrinsic_species (str):
                Extrinsic species - can only deal with one at a time (see
                tutorial on the docs for more complex cases).

        Attributes:
            bulk_composition (str): The bulk (host) composition
            elemental (list): List of elemental species in the bulk composition
            extrinsic_species (str): Extrinsic species, if present
            data (list):
                List of dictionaries containing the parsed competing phases data
            formation_energy_df (pandas.DataFrame):
                DataFrame containing the parsed competing phases data
        """
        self.bulk_composition = Composition(system)
        self.elemental = [str(c) for c in self.bulk_composition.elements]
        self.extrinsic_species = extrinsic_species

        if extrinsic_species:
            self.elemental.append(extrinsic_species)

    # TODO: from_vaspruns and from_csv should be @classmethods so CompetingPhaseAnalyzer can be directly
    #  initialised from them (like Structure.from_file or Distortions.from_structures in SnB etc)
    def from_vaspruns(self, path="competing_phases", folder="vasp_std", csv_path=None, verbose=True):
        """
        Parses competing phase energies from ``vasprun.xml(.gz)`` outputs,
        computes the formation energies and generates the
        ``CompetingPhasesAnalyzer`` object.

        Args:
            path (list, str, pathlib Path):
                Either a path to the base folder in which you have your
                competing phase calculation outputs (e.g.
                formula_EaH_X/vasp_std/vasprun.xml(.gz), or
                formula_EaH_X/vasprun.xml(.gz)), or a list of strings/Paths
                to vasprun.xml(.gz) files.
            folder (str):
                The subfolder in which your vasprun.xml(.gz) output files
                are located (e.g. a file-structure like:
                formula_EaH_X/{folder}/vasprun.xml(.gz)). Default is to
                search for ``vasp_std`` subfolders, or directly in the
                ``formula_EaH_X`` folder.
            csv_path (str):
                If set will save the parsed data to a csv at this filepath.
                Further customisation of the output csv can be achieved with
                the CompetingPhasesAnalyzer.to_csv() method.
            verbose (bool):
                Whether to print out information about directories that were
                skipped (due to no ``vasprun.xml`` files being found).
                Default is ``True``.

        Returns:
            None, sets self.data, self.formation_energy_df and self.elemental_energies
        """
        # TODO: Change this to just recursively search for vaspruns within the specified path (also
        #  currently doesn't seem to revert to searching for vaspruns in the base folder if no vasp_std
        #  subfolders are found) - see how this is done in DefectsParser in analysis.py
        # TODO: Add check for matching INCAR and POTCARs from these calcs - can use code/functions from
        #  analysis.py for this
        self.vasprun_paths = []
        skipped_folders = []

        if isinstance(path, list):  # if path is just a list of all competing phases
            for p in path:
                if "vasprun.xml" in Path(p).name and not Path(p).name.startswith("."):
                    self.vasprun_paths.append(str(Path(p)))

                # try to find the file - will always pick the first match for vasprun.xml*
                elif len(list(Path(p).glob("vasprun.xml*"))) > 0:
                    vsp = next(iter(Path(p).glob("vasprun.xml*")))
                    self.vasprun_paths.append(str(vsp))

                else:
                    skipped_folders.append(p)

        elif isinstance(path, (PurePath, str)):
            path = Path(path)
            for p in path.iterdir():
                if p.is_dir() and not p.name.startswith("."):
                    # add bulk simple properties
                    vr_path = "null_directory"
                    with contextlib.suppress(FileNotFoundError):
                        vr_path, multiple = _get_output_files_and_check_if_multiple(
                            "vasprun.xml", p / folder
                        )
                        if multiple:
                            warnings.warn(
                                f"Multiple `vasprun.xml` files found in directory: {p/folder}. Using "
                                f"{vr_path} to parse the calculation energy and metadata."
                            )

                    if os.path.exists(vr_path):
                        self.vasprun_paths.append(vr_path)

                    else:
                        with contextlib.suppress(FileNotFoundError):
                            vr_path, multiple = _get_output_files_and_check_if_multiple("vasprun.xml", p)
                            if multiple:
                                warnings.warn(
                                    f"Multiple `vasprun.xml` files found in directory: {p}. Using "
                                    f"{vr_path} to parse the calculation energy and metadata."
                                )

                        if os.path.exists(vr_path):
                            self.vasprun_paths.append(vr_path)

                        else:
                            folder_name = str(PurePath(*p.parts[len(path.parts) :]))
                            skipped_folders += [f"{folder_name} or {folder_name}/{folder}"]
        else:
            raise ValueError("Path should either be a list of paths, a string or a pathlib Path object")

        # only warn about skipped folders that are recognised calculation folders (containing a material
        # composition in the name, or 'EaH' in the name)
        skipped_folders_for_warning = []
        for folder_name in skipped_folders:
            comps = []
            for i in folder_name.split(" or ")[0].split("_"):
                with contextlib.suppress(ValueError):
                    comps.append(Composition(i))
            if "EaH" in folder_name or comps:
                skipped_folders_for_warning.append(folder_name)

        if skipped_folders_for_warning and verbose:
            parent_folder_string = f" (in {path})" if isinstance(path, (PurePath, str)) else ""
            warnings.warn(
                f"vasprun.xml files could not be found in the following "
                f"directories{parent_folder_string}, and so they will be skipped for parsing:\n"
                + "\n".join(skipped_folders_for_warning)
            )

        # Ignore POTCAR warnings when loading vasprun.xml
        # pymatgen assumes the default PBE with no way of changing this
        _ignore_pmg_warnings()

        self.vaspruns = []
        failed_parsing_dict = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UnconvergedVASPWarning)  # checked and warned later
            for vasprun_path in tqdm(self.vasprun_paths, desc="Parsing vaspruns..."):
                try:
                    self.vaspruns.append(get_vasprun(vasprun_path))
                except Exception as e:
                    if str(e) in failed_parsing_dict:
                        failed_parsing_dict[str(e)] += [vasprun_path]
                    else:
                        failed_parsing_dict[str(e)] = [vasprun_path]

        if failed_parsing_dict:
            warning_string = (
                "Failed to parse the following `vasprun.xml` files:\n(files: error)\n"
                + "\n".join([f"{paths}: {error}" for error, paths in failed_parsing_dict.items()])
            )
            warnings.warn(warning_string)

        # check if any vaspruns are unconverged, and warn together:
        electronic_unconverged_vaspruns = [
            vr.filename for vr in self.vaspruns if not vr.converged_electronic
        ]
        ionic_unconverged_vaspruns = [vr.filename for vr in self.vaspruns if not vr.converged_ionic]
        for unconverged_vaspruns, unconverged_type in zip(
            [electronic_unconverged_vaspruns, ionic_unconverged_vaspruns],
            ["Electronic", "Ionic"],
        ):
            if unconverged_vaspruns:
                warnings.warn(
                    f"{unconverged_type} convergence was not reached for:\n"
                    + "\n".join(unconverged_vaspruns)
                )

        if not self.vaspruns:
            raise FileNotFoundError(
                "No vasprun files have been parsed, suggesting issues with parsing! Please check that "
                "folders and input parameters are in the correct format (see docstrings/tutorials)."
            )

        data = []
        self.elemental_energies = {}

        for vr in self.vaspruns:
            comp = vr.final_structure.composition
            formulas_per_unit = comp.get_reduced_composition_and_factor()[1]
            energy_per_atom = vr.final_energy / len(vr.final_structure)

            kpoints = (
                "x".join(str(x) for x in vr.kpoints.kpts[0])
                if (vr.kpoints.kpts and len(vr.kpoints.kpts) == 1)
                else "N/A"
            )

            # check if elemental:
            if len(Composition(comp.reduced_formula).as_dict()) == 1:
                el = next(iter(vr.atomic_symbols))  # elemental, so first symbol is only (unique) element
                if el not in self.elemental_energies:
                    self.elemental_energies[el] = energy_per_atom
                    if el not in self.elemental:  # new (extrinsic) element
                        self.elemental.append(el)

                elif energy_per_atom < self.elemental_energies[el]:
                    # only include lowest energy elemental polymorph
                    self.elemental_energies[el] = energy_per_atom

            d = {
                "Formula": comp.reduced_formula,
                "k-points": kpoints,
                "DFT Energy (eV/fu)": vr.final_energy / formulas_per_unit,
                "DFT Energy (eV/atom)": energy_per_atom,
                "DFT Energy (eV)": vr.final_energy,
            }
            data.append(d)

        if self.extrinsic_species and self.extrinsic_species not in self.elemental_energies:
            raise ValueError(
                f"Elemental reference phase for the specified extrinsic species {self.extrinsic_species} "
                f"was not parsed, but is necessary for chemical potential calculations. "
                f"Please ensure that this phase is present in the calculation directory and is being "
                f"correctly parsed."
            )

        formation_energy_df = _calculate_formation_energies(data, self.elemental_energies)
        self.data = formation_energy_df.to_dict(orient="records")
        self.formation_energy_df = pd.DataFrame(self._get_and_sort_formation_energy_data())  # sort data
        self.formation_energy_df.set_index("Formula")

        if csv_path is not None:
            self.to_csv(csv_path)

    def _get_and_sort_formation_energy_data(self, sort_by_energy=False, prune_polymorphs=False):
        data = copy.deepcopy(self.data)

        if prune_polymorphs:  # only keep the lowest energy polymorphs
            formation_energy_df = _calculate_formation_energies(data, self.elemental_energies)
            indices = formation_energy_df.groupby("Formula")["DFT Energy (eV/atom)"].idxmin()
            pruned_df = formation_energy_df.loc[indices]
            data = pruned_df.to_dict(orient="records")

        if sort_by_energy:
            data = sorted(data, key=lambda x: x["Formation Energy (eV/fu)"], reverse=True)

        # moves the bulk composition to the top of the list
        _move_dict_to_start(data, "Formula", self.bulk_composition.reduced_formula)

        # for each dict in data list, sort the keys as formula, formation_energy, energy_per_atom,
        # energy_per_fu, energy, kpoints, then by order of appearance in bulk_composition dict,
        # then alphabetically for any remaining:
        copied_data = copy.deepcopy(data)
        formation_energy_data = [
            {
                **{
                    k: d.pop(k, None)
                    for k in [
                        "Formula",
                        "Formation Energy (eV/fu)",
                        "Formation Energy (eV/atom)",
                        "DFT Energy (eV/atom)",
                        "DFT Energy (eV/fu)",
                        "DFT Energy (eV)",
                        "k-points",
                    ]
                },
                **{  # num elts columns, sorted by order of occurrence in bulk composition:
                    str(elt): d.pop(str(elt), None)
                    for elt in sorted(
                        self.bulk_composition.elements,
                        key=lambda x: self.bulk_composition.reduced_formula.index(str(x)),
                    )
                },
                **{
                    k: v
                    for k, v in d.items()
                    if not any(
                        i in k
                        for i in [
                            "Formula",
                            "Formation Energy",
                            "DFT Energy",
                            "k-points",
                        ]
                    )
                },
            }
            for d in copied_data
        ]
        # if all values are None for a certain key, remove that key from all dicts in list:
        keys_to_remove = [
            k for k in formation_energy_data[0] if all(d[k] is None for d in formation_energy_data)
        ]
        return [{k: v for k, v in d.items() if k not in keys_to_remove} for d in formation_energy_data]

    def to_csv(self, csv_path, sort_by_energy=False, prune_polymorphs=False):
        """
        Write parsed competing phases data to ``csv``. Can be re-loaded with
        ``CompetingPhasesAnalyzer.from_csv()``.

        Args:
            csv_path (str): Path to csv file to write to.
            sort_by_energy (bool):
                If True, sorts the csv by formation energy (highest to lowest).
                Default is False (sorting by formula).
            prune_polymorphs (bool):
                Whether to only write the lowest energy polymorphs for each composition.
                Doesn't affect chemical potential limits (only the ground-state
                polymorphs matter for this).
                Default is False.
        """
        formation_energy_data = self._get_and_sort_formation_energy_data(sort_by_energy, prune_polymorphs)
        pd.DataFrame(formation_energy_data).set_index("Formula").to_csv(csv_path)
        print(f"Competing phase formation energies have been saved to {csv_path}")

    def from_csv(self, csv_path):
        """
        Read in data from a previously parsed formation energies csv file.

        Args:
            csv_path (str): Path to csv file. Must have columns 'Formula',
            and 'DFT Energy per Formula Unit (ev/fu)' or
            'DFT Energy per Atom (ev/atom)'
        Returns:
            None, sets ``self.data`` and ``self.elemental_energies``.
        """
        formation_energy_df = pd.read_csv(csv_path)
        if "Formula" not in list(formation_energy_df.columns) or all(
            x not in list(formation_energy_df.columns)
            for x in [
                "DFT Energy (eV/fu)",
                "DFT Energy (eV/atom)",
            ]
        ):
            raise ValueError(
                "Supplied csv does not contain the minimal columns required ('Formula', and "
                "'DFT Energy (eV/fu)' or 'DFT Energy (eV/atom)')"
            )

        self.data = formation_energy_df.to_dict(orient="records")
        self.elemental_energies = {}
        for i in self.data:
            c = Composition(i["Formula"])
            if len(c.elements) == 1:
                el = c.chemical_system
                if "DFT Energy (eV/atom)" in list(formation_energy_df.columns):
                    el_energy_per_atom = i["DFT Energy (eV/atom)"]
                else:
                    el_energy_per_atom = i["DFT Energy (eV/fu)"] / c.num_atoms

                if el not in self.elemental_energies or el_energy_per_atom < self.elemental_energies[el]:
                    self.elemental_energies[el] = el_energy_per_atom

        if "Formation Energy (eV/fu)" not in list(formation_energy_df.columns):
            formation_energy_df = _calculate_formation_energies(self.data, self.elemental_energies)
            self.data = formation_energy_df.to_dict(orient="records")

        self.formation_energy_df = pd.DataFrame(self._get_and_sort_formation_energy_data())  # sort data
        self.formation_energy_df.set_index("Formula")

    def calculate_chempots(self, csv_path=None, verbose=True, sort_by=None):
        """
        Calculates chemical potential limits. For dopant species, it calculates
        the limiting potential based on the intrinsic chemical potentials (i.e.
        same as ``full_sub_approach=False`` in pycdt).

        Args:
            csv_path (str): If set, will save chemical potential limits to csv
            verbose (bool): If True, will print out chemical potential limits.
            sort_by (str): If set, will sort the chemical potential limits in the output
                dataframe according to the chemical potential of the specified element (from
                element-rich to element-poor conditions).

        Returns:
            Pandas DataFrame, optionally saved to csv.
        """
        intrinsic_phase_diagram_entries = []
        extrinsic_formation_energies = []
        bulk_pde_list = []
        for d in self.data:
            e = PDEntry(d["Formula"], d["DFT Energy (eV/fu)"])
            # checks if the phase is intrinsic
            if set(Composition(d["Formula"]).elements).issubset(self.bulk_composition.elements):
                intrinsic_phase_diagram_entries.append(e)
                if e.composition == self.bulk_composition:  # bulk phase
                    bulk_pde_list.append(e)
            else:
                extrinsic_formation_energies.append(
                    {k: v for k, v in d.items() if k in ["Formula", "Formation Energy (eV/fu)"]}
                )

        if not bulk_pde_list:
            intrinsic_phase_diagram_compositions = (
                {e.composition.reduced_formula for e in intrinsic_phase_diagram_entries}
                if intrinsic_phase_diagram_entries
                else None
            )
            raise ValueError(
                f"Could not find bulk phase for {self.bulk_composition.reduced_formula} in the supplied "
                f"data. Found intrinsic phase diagram entries for: {intrinsic_phase_diagram_compositions}"
            )
        # lowest energy bulk phase
        self.bulk_pde = sorted(bulk_pde_list, key=lambda x: x.energy_per_atom)[0]
        unstable_host = False

        self._intrinsic_phase_diagram = PhaseDiagram(
            intrinsic_phase_diagram_entries,
            map(Element, self.bulk_composition.elements),
        )

        # check if it's stable and if not, warn user and downshift to get _least_ unstable point on convex
        # hull for the host material
        if self.bulk_pde not in self._intrinsic_phase_diagram.stable_entries:
            unstable_host = True
            eah = self._intrinsic_phase_diagram.get_e_above_hull(self.bulk_pde)
            warnings.warn(
                f"{self.bulk_composition.reduced_formula} is not stable with respect to competing phases, "
                f"having an energy above hull of {eah:.4f} eV/atom.\n"
                f"Formally, this means that (based on the supplied athermal calculation data) the host "
                f"material is unstable and so has no chemical potential limits; though in reality the "
                f"host may be stabilised by temperature effects etc, or just a metastable phase.\n"
                f"Here we will determine a single chemical potential 'limit' corresponding to the least "
                f"unstable point on the convex hull for the host material, as an approximation for the "
                f"true chemical potentials."
            )  # TODO: Add example of adjusting the entry energy after loading (if user has calculated
            # e.g. temperature effects) and link in this warning
            # decrease bulk_pde energy per atom by ``e_above_hull`` + 0.1 meV/atom
            renormalised_bulk_pde = _renormalise_entry(self.bulk_pde, eah + 1e-4)
            self._intrinsic_phase_diagram = PhaseDiagram(
                [*intrinsic_phase_diagram_entries, renormalised_bulk_pde],
                map(Element, self.bulk_composition.elements),
            )

        chem_lims = self._intrinsic_phase_diagram.get_all_chempots(self.bulk_composition)

        # remove Element to make it JSONable:
        no_element_chem_lims = {k: {str(kk): vv for kk, vv in v.items()} for k, v in chem_lims.items()}

        if unstable_host:
            no_element_chem_lims = {
                k: {str(kk): vv for kk, vv in v.items()} for k, v in list(chem_lims.items())[:1]
            }

        if sort_by is not None:
            no_element_chem_lims = dict(
                sorted(no_element_chem_lims.items(), key=lambda x: x[1][sort_by], reverse=True)
            )

        self._intrinsic_chempots = {
            "limits": no_element_chem_lims,
            "elemental_refs": {
                str(el): ent.energy_per_atom for el, ent in self._intrinsic_phase_diagram.el_refs.items()
            },
            "limits_wrt_el_refs": {},
        }

        # relate the limits to the elemental energies
        for limit, chempot_dict in self._intrinsic_chempots["limits"].items():
            relative_chempot_dict = copy.deepcopy(chempot_dict)
            for e in relative_chempot_dict:
                relative_chempot_dict[e] -= self._intrinsic_chempots["elemental_refs"][e]
            self._intrinsic_chempots["limits_wrt_el_refs"].update({limit: relative_chempot_dict})

        # get chemical potentials as pandas dataframe
        chemical_potentials = []
        for _, chempot_dict in self._intrinsic_chempots["limits_wrt_el_refs"].items():
            phase_energy_list = []
            phase_name_columns = []
            for k, v in chempot_dict.items():
                phase_name_columns.append(str(k))
                phase_energy_list.append(round(v, 4))
            chemical_potentials.append(phase_energy_list)

        # make df, will need it in next step
        chempots_df = pd.DataFrame(
            chemical_potentials,
            index=list(self._intrinsic_chempots["limits_wrt_el_refs"].keys()),
            columns=phase_name_columns,
        )
        chempots_df.index.name = "Limit"

        if self.extrinsic_species is not None:
            self._calculate_extrinsic_chempot_lims(  # updates self._chempots
                extrinsic_formation_energies=extrinsic_formation_energies,
                chempots_df=chempots_df,
                verbose=verbose,
            )
        else:  # intrinsic only
            self._chempots = self._intrinsic_chempots

        # save and print
        if csv_path is not None:
            chempots_df.to_csv(csv_path)
            if verbose:
                print("Saved chemical potential limits to csv file: ", csv_path)

        if verbose:
            print("Calculated chemical potential limits (in eV wrt elemental reference phases): \n")
            print(chempots_df)

        return chempots_df

    def _calculate_extrinsic_chempot_lims(self, extrinsic_formation_energies, chempots_df, verbose=False):
        if verbose:
            print(f"Calculating chempots for {self.extrinsic_species}")
        for e in extrinsic_formation_energies:
            for el in self.elemental:  # TODO: This code (in all this module) should be rewritten to
                # be more readable (re-used and uninformative variable names, missing informative
                # comments...)
                e[el] = Composition(e["Formula"]).as_dict().get(el, 0)

        # gets the df into a slightly more convenient dict
        cpd = chempots_df.to_dict(orient="records")
        mins = []
        mins_formulas = []
        df3 = pd.DataFrame(extrinsic_formation_energies)
        # print(f"df3: {df3}")  # debugging
        for i, c in enumerate(cpd):
            name = f"mu_{self.extrinsic_species}_{i}"
            df3[name] = df3["Formation Energy (eV/fu)"]
            for k, v in c.items():
                df3[name] -= df3[k] * v
            df3[name] /= df3[self.extrinsic_species]
            # find min at that chempot
            mins.append(df3[name].min())
            mins_formulas.append(df3.iloc[df3[name].idxmin()]["Formula"])

        chempots_df[self.extrinsic_species] = mins
        col_name = f"{self.extrinsic_species}-Limiting Phase"
        chempots_df[col_name] = mins_formulas

        # 1. work out the formation energies of all dopant competing
        #    phases using the elemental energies
        # 2. for each of the chempots already calculated work out what
        #    the chemical potential of the dopant would be from
        #       mu_dopant = Hf(dopant competing phase) - sum(mu_elements)
        # 3. find the most negative mu_dopant which then becomes the new
        #    canonical chemical potential for that dopant species and the
        #    competing phase is the 'limiting phase' right
        # 4. update the chemical potential limits table to reflect this

        # reverse engineer chem lims for extrinsic
        df4 = chempots_df.copy().to_dict(orient="records")
        cl2 = {
            "elemental_refs": self.elemental_energies,
            "limits_wrt_el_refs": {},
            "limits": {},
        }
        # print(f"df4: {df4}")  # debugging

        for i, d in enumerate(df4):
            key = list(self._intrinsic_chempots["limits_wrt_el_refs"].keys())[i] + "-" + d[col_name]
            # print(f"key: {key}")  # debugging
            new_vals = list(self._intrinsic_chempots["limits_wrt_el_refs"].values())[i]
            new_vals[f"{self.extrinsic_species}"] = d[f"{self.extrinsic_species}"]
            cl2["limits_wrt_el_refs"][key] = new_vals
        print(f"cl2: {cl2}")  # debugging

        # relate the limits to the elemental
        # energies but in reverse this time
        for limit, chempot_dict in cl2["limits_wrt_el_refs"].items():
            relative_chempot_dict = copy.deepcopy(chempot_dict)
            for e in relative_chempot_dict:
                relative_chempot_dict[e] += cl2["elemental_refs"][e]
            cl2["limits"].update({limit: relative_chempot_dict})

        self._chempots = cl2

    @property
    def chempots(self) -> dict:
        """
        Returns the calculated chemical potential limits.
        """
        if not hasattr(self, "_chempots"):
            self.calculate_chempots()
        return self._chempots

    @property
    def intrinsic_chempots(self) -> dict:
        """
        Returns the calculated intrinsic chemical potential limits.
        """
        if not hasattr(self, "_intrinsic_chempots"):
            self.calculate_chempots()
        return self._intrinsic_chempots

    @property
    def intrinsic_phase_diagram(self) -> dict:
        """
        Returns the calculated intrinsic phase diagram.
        """
        if not hasattr(self, "_intrinsic_phase_diagram"):
            self.calculate_chempots()
        return self._intrinsic_phase_diagram

    def _cplap_input(self, dependent_variable=None, filename="input.dat"):
        """
        Generates an ``input.dat`` file for the ``CPLAP`` ``FORTRAN`` code
        (legacy code for computing and analysing chemical potential limits, no
        longer recommended).

        Args:
            dependent_variable (str):
                Pick one of the variables as dependent, the first element in
                the composition is chosen if this isn't set.
            filename (str):
                Filename, should end in ``.dat``.

        Returns:
            None, writes input.dat file.
        """
        if not hasattr(self, "chempots"):
            self.calculate_chempots(verbose=False)

        with open(filename, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            # get lowest energy bulk phase
            bulk_entries = [
                sub_dict
                for sub_dict in self.data
                if self.bulk_composition.reduced_composition
                == Composition(sub_dict["Formula"]).reduced_composition
            ]
            bulk_entry = min(bulk_entries, key=lambda x: x["Formation Energy (eV/fu)"])
            print(f"{len(self.bulk_composition.as_dict())}  # number of elements in bulk")
            for k, v in self.bulk_composition.as_dict().items():
                print(int(v), k, end=" ")
            print(
                f"{bulk_entry['Formation Energy (eV/fu)']}  # number of atoms, element, formation "
                f"energy (bulk)"
            )

            if dependent_variable is not None:
                print(f"{dependent_variable}  # dependent variable (element)")
            else:
                print(f"{self.elemental[0]}  # dependent variable (element)")

            # get only the lowest energy entries of compositions in self.data which are on a
            # limit in self._intrinsic_chempots
            bordering_phases = {phase for limit in self._chempots["limits"] for phase in limit.split("-")}
            entries_for_cplap = [
                entry_dict
                for entry_dict in self.data
                if entry_dict["Formula"] in bordering_phases
                and Composition(entry_dict["Formula"]).reduced_composition
                != self.bulk_composition.reduced_composition
            ]
            # cull to only the lowest energy entries of each composition
            culled_cplap_entries = {}
            for entry in entries_for_cplap:
                reduced_comp = Composition(entry["Formula"]).reduced_composition
                if (
                    reduced_comp not in culled_cplap_entries
                    or entry["Formation Energy (eV/fu)"]
                    < culled_cplap_entries[reduced_comp]["Formation Energy (eV/fu)"]
                ):
                    culled_cplap_entries[reduced_comp] = entry

            print(f"{len(culled_cplap_entries)}  # number of bordering phases")
            for i in culled_cplap_entries.values():
                print(f"{len(Composition(i['Formula']).as_dict())}  # number of elements in phase:")
                for k, v in Composition(i["Formula"]).as_dict().items():
                    print(int(v), k, end=" ")
                print(f"{i['Formation Energy (eV/fu)']}  # number of atoms, element, formation energy")

    def to_LaTeX_table(self, splits=1, sort_by_energy=False, prune_polymorphs=True):
        """
        A very simple function to print out the competing phase formation
        energies in a LaTeX table format, showing the formula, kpoints (if
        present in the parsed data) and formation energy.

        Needs the mhchem package to work and does `not` use the booktabs package
        - change hline to toprule, midrule and bottomrule if you want to use
        booktabs style.

        Args:
            splits (int):
                Number of splits for the table; either 1 (default) or 2 (with
                two large columns, each with the formula, kpoints (if present)
                and formation energy (sub-)columns).
            sort_by_energy (bool):
                If True, sorts the table by formation energy (highest to lowest).
                Default is False (sorting by formula).
            prune_polymorphs (bool):
                Whether to only print out the lowest energy polymorphs for each composition.
                Default is True.

        Returns:
            str: LaTeX table string
        """
        if splits not in [1, 2]:
            raise ValueError("`splits` must be either 1 or 2")
        # done in the pyscfermi report style
        formation_energy_data = self._get_and_sort_formation_energy_data(sort_by_energy, prune_polymorphs)

        kpoints_col = any("k-points" in item for item in formation_energy_data)

        string = "\\begin{table}[h]\n\\centering\n"
        string += (
            "\\caption{Formation energies per formula unit ($\\Delta E_f$) of \\ce{"
            + self.bulk_composition.reduced_formula
            + "} and all competing phases"
            + (", with k-meshes used in calculations." if kpoints_col else ".")
            + (" Only the lowest energy polymorphs are included}\n" if prune_polymorphs else "}\n")
        )
        string += "\\label{tab:competing_phase_formation_energies}\n"
        column_names_string = "Formula" + (" & k-mesh" if kpoints_col else "") + " & $\\Delta E_f$ (eV/fu)"

        if splits == 1:
            string += "\\begin{tabular}" + ("{ccc}" if kpoints_col else "{cc}") + "\n"
            string += "\\hline\n"
            string += column_names_string + " \\\\ \\hline \n"
            for i in formation_energy_data:
                kpoints = i.get("k-points", "0x0x0").split("x")
                fe = i["Formation Energy (eV/fu)"]
                string += (
                    "\\ce{"
                    + i["Formula"]
                    + "}"
                    + (f" & {kpoints[0]}$\\times${kpoints[1]}$\\times${kpoints[2]}" if kpoints_col else "")
                    + " & "
                    + f"{fe:.3f} \\\\ \n"
                )

        elif splits == 2:
            string += "\\begin{tabular}" + ("{ccc|ccc}" if kpoints_col else "{cc|cc}") + "\n"
            string += "\\hline\n"
            string += column_names_string + " & " + column_names_string + " \\\\ \\hline \n"

            mid = len(formation_energy_data) // 2
            first_half = formation_energy_data[:mid]
            last_half = formation_energy_data[mid:]

            for i, j in zip(first_half, last_half):
                kpoints1 = i.get("k-points", "0x0x0").split("x")
                fe1 = i["Formation Energy (eV/fu)"]
                kpoints2 = j.get("k-points", "0x0x0").split("x")
                fe2 = j["Formation Energy (eV/fu)"]
                string += (
                    "\\ce{"
                    + i["Formula"]
                    + "}"
                    + (
                        f" & {kpoints1[0]}$\\times${kpoints1[1]}$\\times${kpoints1[2]}"
                        if kpoints_col
                        else ""
                    )
                    + " & "
                    + f"{fe1:.3f} & "
                    + "\\ce{"
                    + j["Formula"]
                    + "}"
                    + (
                        f" & {kpoints2[0]}$\\times${kpoints2[1]}$\\times${kpoints2[2]}"
                        if kpoints_col
                        else ""
                    )
                    + " & "
                    + f"{fe2:.3f} \\\\ \n"
                )

        string += "\\hline\n"
        string += "\\end{tabular}\n"
        string += "\\end{table}"

        return string


def get_X_rich_limit(X: str, chempots: dict):
    """
    Determine the chemical potential limit of the input chempots dict which
    corresponds to the most X-rich conditions.

    Args:
        X (str): Elemental species (e.g. "Te")
        chempots (dict):
            The chemical potential limits dict, as returned by
            ``CompetingPhasesAnalyzer.chempots``
    """
    X_rich_limit = None
    X_rich_limit_chempot = None
    for limit, chempot_dict in chempots["limits"].items():
        if X in chempot_dict and (X_rich_limit is None or chempot_dict[X] > X_rich_limit_chempot):
            X_rich_limit = limit
            X_rich_limit_chempot = chempot_dict[X]

    if X_rich_limit is None:
        raise ValueError(f"Could not find {X} in the chemical potential limits dict:\n{chempots}")

    return X_rich_limit


def get_X_poor_limit(X: str, chempots: dict):
    """
    Determine the chemical potential limit of the input chempots dict which
    corresponds to the most X-poor conditions.

    Args:
        X (str): Elemental species (e.g. "Te")
        chempots (dict):
            The chemical potential limits dict, as returned by
            ``CompetingPhasesAnalyzer.chempots``
    """
    X_poor_limit = None
    X_poor_limit_chempot = None
    for limit, chempot_dict in chempots["limits"].items():
        if X in chempot_dict and (X_poor_limit is None or chempot_dict[X] < X_poor_limit_chempot):
            X_poor_limit = limit
            X_poor_limit_chempot = chempot_dict[X]

    if X_poor_limit is None:
        raise ValueError(f"Could not find {X} in the chemical potential limits dict:\n{chempots}")

    return X_poor_limit


def _move_dict_to_start(data, key, value):
    for index, item in enumerate(data):
        if key in item and item[key] == value:
            data.insert(0, data.pop(index))
            return


def combine_extrinsic(first, second, extrinsic_species):
    # TODO: Can we just integrate this to `CompetingPhaseAnalyzer`, so you just pass in a list of
    # extrinsic species and it does the right thing?
    """
    Combines chemical limits for different extrinsic species using chemical
    limits json file from ChemicalPotentialAnalysis.

    Usage explained in the example jupyter notebook
    Args:
        first (dict): First chemical potential dictionary, it can contain extrinsic species other
        than the set extrinsic species
        second (dict): Second chemical potential dictionary, it must contain the extrinsic species
        extrinsic_species (str): Extrinsic species in the second dictionary
    Returns:
        dict.
    """
    keys = ["elemental_refs", "limits", "limits_wrt_el_refs"]
    if any(key not in first for key in keys):
        raise KeyError(
            "the first dictionary doesn't contain the correct keys - it should include "
            "elemental_refs, limits and limits_wrt_el_refs"
        )

    if any(key not in second for key in keys):
        raise KeyError(
            "the second dictionary doesn't contain the correct keys - it should include "
            "elemental_refs, limits and limits_wrt_el_refs"
        )

    if extrinsic_species not in second["elemental_refs"]:
        raise ValueError("extrinsic species is not present in the second dictionary")

    cpa1 = copy.deepcopy(first)
    cpa2 = copy.deepcopy(second)
    new_limits = {}
    for (k1, v1), (k2, v2) in zip(list(cpa1["limits"].items()), list(cpa2["limits"].items())):
        if k2.rsplit("-", 1)[0] in k1:
            new_key = k1 + "-" + k2.rsplit("-", 1)[1]
        else:
            raise ValueError("The limits aren't matching, make sure you've used the correct dictionary")

        v1[extrinsic_species] = v2.pop(extrinsic_species)
        new_limits[new_key] = v1

    new_limits_wrt_el = {}
    for (k1, v1), (k2, v2) in zip(
        list(cpa1["limits_wrt_el_refs"].items()),
        list(cpa2["limits_wrt_el_refs"].items()),
    ):
        if k2.rsplit("-", 1)[0] in k1:
            new_key = k1 + "-" + k2.rsplit("-", 1)[1]
        else:
            raise ValueError("The limits aren't matching, make sure you've used the correct dictionary")

        v1[extrinsic_species] = v2.pop(extrinsic_species)
        new_limits_wrt_el[new_key] = v1

    new_elements = copy.deepcopy(cpa1["elemental_refs"])
    new_elements[extrinsic_species] = copy.deepcopy(cpa2["elemental_refs"])[extrinsic_species]

    return {
        "elemental_refs": new_elements,
        "limits": new_limits,
        "limits_wrt_el_refs": new_limits_wrt_el,
    }
