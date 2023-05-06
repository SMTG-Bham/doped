"""
This module contains calls to finite size supercell charge corrections as implemented in
pymatgen.analysis.defects.core

The methods implemented are:
1) Freysoldt correction for isotropic systems. Includes:
       a) PC energy
       b) potential alignment by planar averaging.
2) Extended Freysoldt or Kumagai correction for anistropic systems. Includes:
       a) anisotropic PC energy
       b) potential alignment by atomic site averaging outside Wigner Seitz radius

If you use the corrections implemented in this module, cite
   Freysoldt, Neugebauer, and Van de Walle,
    Phys. Status Solidi B. 248, 1067-1076 (2011) for isotropic correction
   Kumagai and Oba, Phys. Rev. B. 89, 195205 (2014) for anisotropic correction
"""


import copy

import numpy as np
from pymatgen.analysis.defects.corrections import FreysoldtCorrection, KumagaiCorrection
from monty.json import MontyDecoder

from doped.pycdt.utils.parse_calculations import _convert_dielectric_to_tensor


def _monty_decode_nested_dicts(d):
    """
    Recursively find any dictionaries in defect_entry.parameters, which may be nested in dicts or in
    lists of dicts and decode them:
    """
    for key, value in d.items():
        if isinstance(value, dict) and not any(
            k in value for k in ["@module", "@class"]
        ):
            _monty_decode_nested_dicts(value)
        elif isinstance(value, list):
            if all(isinstance(i, dict) for i in value) and all(
                k in i for k in ["@module", "@class"] for i in value
            ):
                try:
                    d[key] = [MontyDecoder().process_decoded(i) for i in value]
                except Exception as exc:
                    print(f"Failed to decode {key} with error {exc}")
                    pass

        if isinstance(value, dict) and all(k in value for k in ["@module", "@class"]):
            try:
                d[key] = MontyDecoder().process_decoded(value)
            except Exception as exc:
                print(f"Failed to decode {key} with error {exc}")
                pass


def get_correction_freysoldt(
    defect_entry,
    epsilon,
    plot: bool = False,
    filename: str = None,
    partflag="All",
    axis=None,
):
    """
    Function to compute the isotropic Freysoldt correction for each defect.
    If this correction is used, please reference Freysoldt's original paper.
    doi: 10.1103/PhysRevLett.102.016402
    Args:
        defect_entry: DefectEntry object with the following
            keys stored in defect.parameters:
                required:
                    axis_grid (3 x NGX where NGX is the length of the NGX grid
                    in the x,y and z axis directions. Same length as planar
                    average lists):
                        A list of 3 numpy arrays which contain the cartesian axis
                        values (in angstroms) that correspond to each planar avg
                        potential supplied.

                    bulk_planar_averages (3 x NGX where NGX is the length of
                    the NGX grid in the x,y and z axis directions.):
                        A list of 3 numpy arrays which contain the planar averaged
                        electrostatic potential for the bulk supercell.

                    defect_planar_averages (3 x NGX where NGX is the length of
                    the NGX grid in the x,y and z axis directions.):
                        A list of 3 numpy arrays which contain the planar averaged
                        electrostatic potential for the defective supercell.

                    bulk_sc_structure (Structure) bulk structure corresponding to
                        defect supercell structure (uses Lattice for charge correction)

                    defect_frac_sc_coords (3 x 1 array) Fracitional co-ordinates of
                        defect location in supercell structure
                optional:
                    'encut' : energy cutoff desired for Freysoldt correction
                    'madetol' : madelung tolerance for Freysoldt correction
                    'q_model' : Charge Model for Freysoldt correction
                    'q_model' : Charge Model for Freysoldt correction
        epsilon (float or int or 3x1 matrix or 3x3 matrix):
            ionic + static contributions to dielectric constant
        plot (bool): decides whether to plot electrostatic potential plots or not.
        filename (str): if None, plots are not saved, if a string,
            then the plot will be saved as '{filename}_{axis}.pdf'
        partflag: four options for correction output:
               'pc' for just point charge correction, or
               'potalign' for just potalign correction, or
               'All' for both (added together), or
               'AllSplit' for individual parts split up (form is [PC, potterm, full])
        axis (int or None): if integer, then freysoldt correction is performed on the single axis.
            If it is None, then averaging of the corrections for the three axes is used for the
            correction.

    Returns Correction
    """
    # ensure parameters are decoded in case defect_dict was reloaded from json
    _monty_decode_nested_dicts(defect_entry.parameters)

    epsilon = _convert_dielectric_to_tensor(epsilon)

    if partflag not in ["All", "AllSplit", "pc", "potalign"]:
        print(
            '{} is incorrect potalign type. Must be "All", "AllSplit", "pc", or '
            '"potalign".'.format(partflag)
        )
        return

    q_model = defect_entry.parameters.get("q_model", None)
    encut = defect_entry.parameters.get("encut", 520)
    madetol = defect_entry.parameters.get("madetol", 0.0001)

    if not defect_entry.charge:
        print("Charge is zero so charge correction is zero.")
        return 0.0

    template_defect = copy.deepcopy(defect_entry)
    corr_class = FreysoldtCorrection(
        epsilon, q_model=q_model, energy_cutoff=encut, madetol=madetol, axis=axis
    )
    f_corr_summ = corr_class.get_correction(template_defect)

    if plot:
        if axis is None:
            ax_list = [
                [k, "axis" + str(k)]
                for k in corr_class.metadata["pot_plot_data"].keys()
            ]
        else:
            ax_list = [[axis, "axis" + str(axis + 1)]]

        for ax_key, ax_title in ax_list:
            p = corr_class.plot(ax_key, title=ax_title, saved=False)
            if filename:
                p.savefig(filename + "_" + ax_title + ".pdf", bbox_inches="tight")

    if partflag in ["AllSplit", "All"]:
        freyval = np.sum(list(f_corr_summ.values()))
    elif partflag == "pc":
        freyval = f_corr_summ["freysoldt_electrostatic"]
    elif partflag == "potalign":
        freyval = f_corr_summ["freysoldt_potential_alignment"]

    print(f"Final Freysoldt correction is {freyval:.3f} eV")

    if partflag == "AllSplit":
        freyval = [
            f_corr_summ["freysoldt_electrostatic"],
            f_corr_summ["freysoldt_potential_alignment"],
            freyval,
        ]

    return freyval


def get_correction_kumagai(
    defect_entry, epsilon, plot: bool = False, filename: str = None, partflag="All"
):
    """
    Function to compute the Kumagai correction for each defect (modified freysoldt for
    anisotropic dielectric).
    NOTE that bulk_init class must be pre-instantiated to use this function
    Args:
        defect_entry: DefectEntry object with the following
            keys stored in defect.parameters:
                required:
                    bulk_atomic_site_averages (list):  list of bulk structure"s atomic site
                    averaged ESPs * charge, in same order as indices of bulk structure note this
                    is list given by VASP's OUTCAR (so it is multiplied by a test charge of -1)

                    defect_atomic_site_averages (list):  list of defect structure"s atomic site
                    averaged ESPs * charge, in same order as indices of defect structure note
                    this is list given by VASP's OUTCAR (so it is multiplied by a test charge of -1)

                    site_matching_indices (list):  list of corresponding site index values for
                    bulk and defect site structures EXCLUDING the defect site itself (ex. [[bulk
                    structure site index, defect structure"s corresponding site index], ... ]

                    initial_defect_structure (Structure): Pymatgen Structure object representing
                    un-relaxed defect structure

                    defect_frac_sc_coords (array): Defect Position in fractional coordinates of
                    the supercell given in bulk_structure
                optional:
                    gamma (float): Ewald parameter, Default is to determine it based on
                        convergence of brute summation tolerance
                    sampling_radius (float): radius (in Angstrom) which sites must be outside of
                        to be included in the correction. Publication by Kumagai advises to use
                        Wigner-Seitz radius of defect supercell, so this is default value.
        epsilon (float or int or 3x1 matrix or 3x3 matrix):
            ionic + static contributions to dielectric constant
        plot (bool): decides whether to plot electrostatic potential plots or not.
        filename (str): if None, plots are not saved, if a string, then the plot will be saved as
            '{filename}.pdf'
        partflag: four options for correction output:
               'pc' for just point charge correction, or
               'potalign' for just potalign correction, or
               'All' for both (added together), or
               'AllSplit' for individual parts split up (form is [PC, potterm, full])
    """
    # ensure parameters are decoded in case defect_dict was reloaded from json
    _monty_decode_nested_dicts(defect_entry.parameters)

    epsilon = _convert_dielectric_to_tensor(epsilon)

    if partflag not in ["All", "AllSplit", "pc", "potalign"]:
        print(
            '{} is incorrect potalign type. Must be "All", "AllSplit", "pc", or '
            '"potalign".'.format(partflag)
        )
        return

    sampling_radius = defect_entry.parameters.get("sampling_radius", None)
    gamma = defect_entry.parameters.get("gamma", None)

    if not defect_entry.charge:
        print("Charge is zero so charge correction is zero.")
        return 0.0

    template_defect = copy.deepcopy(defect_entry)
    corr_class = KumagaiCorrection(
        epsilon, sampling_radius=sampling_radius, gamma=gamma
    )
    k_corr_summ = corr_class.get_correction(template_defect)

    if plot:
        p = corr_class.plot(title="Kumagai", saved=False)
        if filename:
            p.savefig(f"{filename}.pdf", bbox_inches="tight")

    if partflag in ["AllSplit", "All"]:
        kumagai_val = np.sum(list(k_corr_summ.values()))
    elif partflag == "pc":
        kumagai_val = k_corr_summ["kumagai_electrostatic"]
    elif partflag == "potalign":
        kumagai_val = k_corr_summ["kumagai_potential_alignment"]

    print(f"\nFinal Kumagai correction is {kumagai_val:.3f} eV")

    if partflag == "AllSplit":
        kumagai_val = [
            k_corr_summ["kumagai_electrostatic"],
            k_corr_summ["kumagai_potential_alignment"],
            kumagai_val,
        ]
    return kumagai_val
