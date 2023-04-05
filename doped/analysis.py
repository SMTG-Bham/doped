# coding: utf-8

"""
Code to analyse VASP defect calculations.
These functions are built from a combination of useful modules from pymatgen, alongside
substantial modification, in the efforts of making an efficient, user-friendly package for
managing and analysing defect calculations, with publication-quality outputs
"""

import copy
from operator import itemgetter

import pandas as pd
from pymatgen.analysis.defects.thermodynamics import DefectPhaseDiagram
from pymatgen.util.string import unicodeify
from tabulate import tabulate

from doped import aide_murphy_correction


def bold_print(string: str) -> None:
    """Does what it says on the tin. Prints the input string in bold."""
    print("\033[1m" + string + "\033[0m")


def dpd_from_parsed_defect_dict(parsed_defect_dict: dict) -> DefectPhaseDiagram:
    """Generates a DefectPhaseDiagram object from a dictionary of parsed defect calculations (
    format: {"defect_name": defect_entry}), likely created using SingleDefectParser from
    doped.pycdt.utils.parse_calculations), which can then be used to analyse and plot the defect
    thermodynamics (formation energies, transition levels, concentrations etc)

    Args:
        parsed_defect_dict (dict):
            Dictionary of parsed defect calculations (format: {"defect_name": defect_entry}),
            likely created using SingleDefectParser from doped.pycdt.utils.parse_calculations).
            Must have 'vbm' and 'gap' in defect_entry.parameters for each defect (from
            SingleDefectParser.get_bulk_gap_data())

    Returns:
        pymatgen DefectPhaseDiagram object (DefectPhaseDiagram)
    """
    vbm_vals = []
    bandgap_vals = []
    for defect in parsed_defect_dict.values():
        vbm_vals.append(defect.parameters["vbm"])
        bandgap_vals.append(defect.parameters["gap"])
    if len(set(vbm_vals)) > 1:  # Check if all defects give same vbm
        raise ValueError(
            f"VBM values don't match for defects in given defect dictionary, "
            f"the VBM values in the dictionary are: {vbm_vals}. "
            f"Are you sure the correct/same bulk files were used with "
            f"SingleDefectParser and/or get_bulk_gap_data()?"
        )
    if len(set(bandgap_vals)) > 1:  # Check if all defects give same bandgap
        raise ValueError(
            f"Bandgap values don't match for defects in given defect dictionary, "
            f"the bandgap values in the dictionary are: {bandgap_vals}. "
            f"Are you sure the correct/same bulk files were used with "
            f"SingleDefectParser and/or get_bulk_gap_data()?"
        )
    vbm = vbm_vals[0]
    bandgap = bandgap_vals[0]
    dpd = DefectPhaseDiagram(
        list(parsed_defect_dict.values()), vbm, bandgap, filter_compatible=False
    )

    return dpd


def dpd_transition_levels(defect_phase_diagram: DefectPhaseDiagram):
    """Iteratively prints the charge transition levels for the input DefectPhaseDiagram object
    (via the from a defect_phase_diagram.transition_level_map attribute)

    Args:
        defect_phase_diagram (DefectPhaseDiagram):
            DefectPhaseDiagram object (likely created from
            analysis.dpd_from_parsed_defect_dict)

    Returns:
        None
    """
    for def_type, tl_info in defect_phase_diagram.transition_level_map.items():
        bold_print(f"\nDefect: {def_type.split('@')[0]}")
        for tl_efermi, chargeset in tl_info.items():
            print(
                f"Transition Level ({max(chargeset):{'+' if max(chargeset) else ''}}/"
                f"{min(chargeset):{'+' if min(chargeset) else ''}}) at {tl_efermi:.3f}"
                f" eV above the VBM"
            )


def formation_energy_table(
    defect_phase_diagram: DefectPhaseDiagram,
    chempot_limits: dict = None,
    pd_facets: list = None,
    fermi_level: float = 0,
    hide_cols: list = None,
    show_key: bool = True,
):
    """
    Prints defect formation energy tables for either a single chemical potential limit (i.e. phase
    diagram facet) or each facet in the phase diagram (chempot_limits dict), depending on the
    chempot_limits input supplied. This can either be a dictionary of chosen absolute/DFT chemical
    potentials: {Elt: Energy} (giving a single formation energy table) or a dictionary including
    the key-value pair: {"facets": [{'facet': [chempot_dict]}]}, following the format generated
    by chempot_limits = cpa.read_phase_diagram_and_chempots() (see example notebooks). In the
    latter case, a subset of facet(s) / chemical potential limit(s) can be chosen with the
    pd_facets argument, or if not specified, will print formation energy tables for each facet in
    the phase diagram.
    Returns the results a pandas DataFrame or list of DataFrames.

    Args:
        defect_phase_diagram (DefectPhaseDiagram):
             DefectPhaseDiagram object (likely created from
             analysis.dpd_from_parsed_defect_dict)
        chempot_limits (dict):
            This can either be a dictionary of chosen absolute/DFT chemical potentials: {Elt:
            Energy} (giving a single formation energy table) or a dictionary including the
            key-value pair: {"facets": [{'facet': [chempot_dict]}]}, following the format generated
            by chempot_limits = cpa.read_phase_diagram_and_chempots() (see example notebooks). If
            not specified, chemical potentials are not included in the formation energy calculation
            (all set to zero energy).
        pd_facets (list):
            A list facet(s) / chemical potential limit(s) for which to print the defect formation
            energy tables. If not specified, will print formation energy tables for each facet in
            the phase diagram. (default: None)
        fermi_level (float):
            Fermi level to use for computing the defect formation energies. (default: 0 (i.e.
            at the VBM))
        hide_cols: (list):
            List of columns to hide from the output. (default: None)
        show_key (bool):
            Whether or not to print the table key at the bottom of the output. (default: True)

    Returns:
        pandas DataFrame or list of DataFrames
    """
    if chempot_limits is None:
        chempot_limits = {}

    if "facets" in chempot_limits:
        list_of_dfs = []
        if not pd_facets:
            pd_facets = chempot_limits[
                "facets"
            ].keys()  # Phase diagram facets to use for chemical
            # potentials, to tabulate formation energies
        for facet in pd_facets:
            bold_print("Facet: " + unicodeify(facet))
            df = single_formation_energy_table(
                defect_phase_diagram,
                chempots=chempot_limits["facets"][facet],
                fermi_level=fermi_level,
                hide_cols=hide_cols,
                show_key=show_key,
            )
            list_of_dfs.append(df)
            print("\n")

        return list_of_dfs

    # else return {Elt: Energy} dict for chempot_limits, or if unspecified, all zero energy
    df = single_formation_energy_table(
        defect_phase_diagram,
        chempots=chempot_limits,
        fermi_level=fermi_level,
        hide_cols=hide_cols,
        show_key=show_key,
    )
    return df


def single_formation_energy_table(
    defect_phase_diagram: DefectPhaseDiagram,
    chempots: dict = None,
    fermi_level: float = 0,
    hide_cols: list = None,
    show_key: bool = True,
):
    """
    Prints a defect formation energy table for a single chemical potential limit (i.e. phase diagram
    facet), and returns the results as a pandas DataFrame.

    Args:
        defect_phase_diagram (DefectPhaseDiagram):
             DefectPhaseDiagram object (likely created from
             analysis.dpd_from_parsed_defect_dict)
        chempots (dict):
            Dictionary of chosen absolute/DFT chemical potentials: {Elt: Energy}. If not
            specified, chemical potentials are not included in the formation energy calculation
            (all set to zero energy).
        fermi_level (float):
            Fermi level to use for computing the defect formation energies. (default: 0 (i.e.
            at the VBM))
        hide_cols: (list):
            List of columns to hide from the output. (default: None)
        show_key (bool):
            Whether or not to print the table key at the bottom of the output. (default: True)

    Returns:
        pandas DataFrame sorted by formation energy
    """
    header = ["Defect", "Charge", "Defect Path"]
    table = []
    if hide_cols is None:
        hide_cols = []

    for defect_entry in defect_phase_diagram.entries:
        row = [
            defect_entry.name,
            defect_entry.charge,
            defect_entry.parameters["defect_path"],
        ]
        if "Uncorrected Energy" not in hide_cols:
            header += ["Uncorrected Energy"]
            row += [f"{defect_entry.uncorrected_energy:.2f} eV"]
        if "Corrected Energy" not in hide_cols:
            header += ["Corrected Energy"]
            row += [
                f"{defect_entry.energy:.2f} eV"
            ]  # With 0 chemical potentials, at the calculation
            # fermi level
        header += ["Formation Energy"]
        formation_energy = defect_entry.formation_energy(
            chemical_potentials=chempots, fermi_level=fermi_level
        )
        row += [f"{formation_energy:.2f} eV"]

        table.append(row)
    table = sorted(table, key=itemgetter(0, 1))
    print(
        tabulate(
            table,
            headers=header,
            tablefmt="fancy_grid",
            stralign="left",
            numalign="left",
        ),
        "\n",
    )

    if show_key:
        bold_print("Table Key:")
        print(
            """'Defect' -> Defect Type and Multiplicity
'Charge' -> Defect Charge State
'Uncorrected Energy' -> Defect Energy from calculation, without corrections
'Corrected Energy' -> Defect Energy from calculation (E_defect - E_host + corrections)
(chemical potentials set to 0 and the fermi level at average electrostatic potential in the
supercell)
'Formation Energy' -> Final Defect Formation Energy, with the specified chemical potentials (
chempot_limits)(default: all 0) and the chosen fermi_level (default: 0)(i.e. at the VBM)
        """
        )

    sorted_df = pd.DataFrame(
        table,
        columns=[
            "Defect",
            "Charge",
            "Defect Path",
            "Uncorrected Energy",
            "Corrected Energy",
            "Formation Energy",
        ],
    )
    sorted_df = sorted_df.sort_values("Formation Energy")
    return sorted_df


def lany_zunger_corrected_defect_dict_from_freysoldt(defect_dict: dict):
    """Convert input parsed defect dictionary (presumably created using SingleDefectParser
    from doped.pycdt.utils.parse_calculations) with Freysoldt charge corrections to
    the same parsed defect dictionary but with the Lany-Zunger charge correction (same potential
    alignment plus 0.65 * Makov-Payne image charge correction (same image charge correction as
    Freysoldt scheme)).
    Args:
        parsed_defect_dict (dict):
            Dictionary of parsed defect calculations (presumably created using SingleDefectParser
            from doped.pycdt.utils.parse_calculations) (see example notebook)
            Must have 'freysoldt_meta' in defect.parameters for each charged defect (from
            SingleDefectParser.freysoldt_loader())
    Returns:
        Parsed defect dictionary with Lany-Zunger charge corrections.
    """
    random_defect_entry = list(defect_dict.values())[
        0
    ]  # Just need any DefectEntry from
    # defect_dict to get the lattice and dielectric matrix
    lattice = random_defect_entry.bulk_structure.lattice.matrix
    dielectric = random_defect_entry.parameters["dielectric"]
    lz_image_charge_corrections = aide_murphy_correction.get_image_charge_correction(
        lattice, dielectric
    )
    lz_corrected_defect_dict = copy.deepcopy(defect_dict)
    for defect_name, defect_entry in lz_corrected_defect_dict.items():
        if defect_entry.charge != 0:
            potalign = defect_entry.parameters["freysoldt_meta"][
                "freysoldt_potential_alignment_correction"
            ]
            mp_pc_corr = lz_image_charge_corrections[
                abs(defect_entry.charge)
            ]  # Makov-Payne PC correction
            defect_entry.parameters.update(
                {
                    "Lany-Zunger_Corrections": {
                        "(Freysoldt)_Potential_Alignment_Correction": potalign,
                        "Makov-Payne_Image_Charge_Correction": mp_pc_corr,
                        "Lany-Zunger_Scaled_Image_Charge_Correction": 0.65 * mp_pc_corr,
                        "Total_Lany-Zunger_Correction": potalign + 0.65 * mp_pc_corr,
                    }
                }
            )
            defect_entry.corrections["charge_correction"] = defect_entry.parameters[
                "Lany-Zunger_Corrections"
            ]["Total_Lany-Zunger_Correction"]

        lz_corrected_defect_dict.update({defect_name: defect_entry})
    return lz_corrected_defect_dict


def lany_zunger_corrected_defect_dict_from_kumagai(defect_dict: dict):
    """Convert input parsed defect dictionary (presumably created using SingleDefectParser
    from doped.pycdt.utils.parse_calculations) with Kumagai charge corrections to
    the same parsed defect dictionary but with the 'Lany-Zunger' charge correction (same potential
    alignment plus 0.65 * image charge correction.
    Args:
        parsed_defect_dict (dict):
            Dictionary of parsed defect calculations (presumably created using SingleDefectParser
            from doped.pycdt.utils.parse_calculations) (see example notebook)
            Must have 'kumagai_meta' in defect.parameters for each charged defect (from
            SingleDefectParser.kumagai_loader())
    Returns:
        Parsed defect dictionary with Lany-Zunger charge corrections.
    """
    random_defect_entry = list(defect_dict.values())[
        0
    ]  # Just need any DefectEntry from
    # defect_dict to get the lattice and dielectric matrix
    lattice = random_defect_entry.bulk_structure.lattice.matrix
    dielectric = random_defect_entry.parameters["dielectric"]
    lz_image_charge_corrections = aide_murphy_correction.get_image_charge_correction(
        lattice, dielectric
    )
    lz_corrected_defect_dict = copy.deepcopy(defect_dict)
    for defect_name, defect_entry in lz_corrected_defect_dict.items():
        if defect_entry.charge != 0:
            potalign = defect_entry.parameters["kumagai_meta"][
                "kumagai_potential_alignment_correction"
            ]
            makove_payne_pc_correction = lz_image_charge_corrections[
                abs(defect_entry.charge)
            ]
            defect_entry.parameters.update(
                {
                    "Lany-Zunger_Corrections": {
                        "(Kumagai)_Potential_Alignment_Correction": potalign,
                        "Makov-Payne_Image_Charge_Correction": makove_payne_pc_correction,
                        "Lany-Zunger_Scaled_Image_Charge_Correction": 0.65
                        * makove_payne_pc_correction,
                        "Total_Lany-Zunger_Correction": potalign
                        + 0.65 * makove_payne_pc_correction,
                    }
                }
            )
            defect_entry.corrections["charge_correction"] = defect_entry.parameters[
                "Lany-Zunger_Corrections"
            ]["Total_Lany-Zunger_Correction"]

        lz_corrected_defect_dict.update({defect_name: defect_entry})
    return lz_corrected_defect_dict
