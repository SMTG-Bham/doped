"""
Code to generate VASP defect calculation input files.
"""
import os
import warnings
import numpy as np
from copy import deepcopy  # See https://stackoverflow.com/a/22341377/14020960
from typing import TYPE_CHECKING, Optional, Union, List, Dict

from monty.io import zopen
from monty.serialization import dumpfn, loadfn
from pymatgen.core import SETTINGS
from pymatgen.io.vasp.inputs import (
    BadIncarWarning,
    incar_params,
    Incar,
    Kpoints,
    Poscar,
    Potcar,
)
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.sets import DictSet
from pymatgen.analysis.defects.thermo import DefectEntry
from shakenbreak.input import (
    _get_defect_site,
    _get_bulk_defect_site,
    _get_defect_name_from_obj,
    _update_defect_dict,
)

from doped import _ignore_pmg_warnings
from doped.generation import DefectsGenerator

if TYPE_CHECKING:
    import pymatgen.core.periodic_table
    import pymatgen.core.structure

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
default_potcar_dict = loadfn(os.path.join(MODULE_DIR, "PotcarSet.yaml"))
default_relax_set = loadfn(os.path.join(MODULE_DIR, "HSE06_RelaxSet.yaml"))
default_defect_set = loadfn(os.path.join(MODULE_DIR, "DefectSet.yaml"))
default_relax_set["INCAR"].update(default_defect_set["INCAR"])

_ignore_pmg_warnings()


class DefectRelaxSet(DictSet):
    def __init__(
        self,
        structure: Structure,
        config_dict: Optional[Dict] = None,
        charge: int = 0,
        poscar_comment: Optional[str] = None,
        **kwargs,
    ):
        """
        Extension to pymatgen DictSet object for VASP Defect Relaxation Calculations

        Args:
            structure (Structure): pymatgen Structure object of the defect supercell
            config_dict (dict): The config dictionary to use.
            charge (int): Charge of the defect structure
            poscar_comment (str): POSCAR file comment
            **kwargs: Additional kwargs to pass to DictSet
        """
        self.charge = charge
        self.poscar_comment = poscar_comment

        if config_dict is not None:
            self.CONFIG = config_dict  # Bug in pymatgen 2023.5.10 # TODO: PR pymatgen to fix this Yb issue with `DictSet`
            super(self.__class__, self).__init__(
                structure, config_dict=config_dict, **kwargs
            )
        else:
            MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
            default_potcar_dict = loadfn(os.path.join(MODULE_DIR, "PotcarSet.yaml"))
            relax_set = loadfn(os.path.join(MODULE_DIR, "HSE06_RelaxSet.yaml"))
            defect_set = loadfn(os.path.join(MODULE_DIR, "DefectSet.yaml"))
            relax_set["INCAR"].update(defect_set["INCAR"])
            relax_set.update(default_potcar_dict)

            from doped.vasp import scaled_ediff

            relax_set["INCAR"]["EDIFF"] = scaled_ediff(len(structure))

            self.CONFIG = relax_set
            super(self.__class__, self).__init__(
                structure, config_dict=relax_set, **kwargs
            )

    @property
    def incar(self):
        inc = super(self.__class__, self).incar
        try:
            inc["NELECT"] = self.nelect - self.charge
            if inc["NELECT"] % 2 != 0:  # odd number of electrons
                inc["NUPDOWN"] = 1
            else:
                # when writing VASP just resets this to 0 anyway:
                inc["NUPDOWN"] = (
                    "0  # If defect has multiple spin-polarised states (e.g. bipolarons) could "
                    "also have triplet (NUPDOWN=2), but ΔE typically small."
                )

        except Exception as e:
            warnings.warn(
                f"NELECT and NUPDOWN flags are not set due to non-availability of POTCARs; "
                f"got error {e}"
            )

        return inc

    @property
    def potcar(self):
        """
        Potcar object.
        """
        return Potcar(symbols=self.potcar_symbols, functional=self.potcar_functional)

    @property
    def poscar(self) -> Poscar:
        """
        Return Poscar object with comment
        """
        return Poscar(self.structure, comment=self.poscar_comment)

    @property
    def all_input(self):
        """
        Returns all input files as a dict of {filename: vasp object}

        Returns:
            dict of {filename: object}, e.g., {'INCAR': Incar object, ...}
        """
        try:
            return super(DefectRelaxSet, self).all_input
        except:  # Expecting the error to be POTCAR related, its ignored
            kpoints = self.kpoints
            incar = self.incar
            if np.product(kpoints.kpts) < 4 and incar.get("ISMEAR", 0) == -5:
                incar["ISMEAR"] = 0

            return {"INCAR": incar, "KPOINTS": kpoints, "POSCAR": self.poscar}

    @property
    def structure(self):
        """
        :return: Structure
        """
        return self._structure


def scaled_ediff(natoms: int) -> float:
    """
    Returns a scaled EDIFF value for VASP calculations, based on the number of atoms in the
    structure. EDIFF is set to 1e-5 per 50 atoms in the supercell, with a maximum EDIFF of 1e-4.
    """
    ediff = float(f"{((natoms/50)*1e-5):.1g}")
    return ediff if ediff <= 1e-4 else 1e-4


def _get_transformation_info(defect_species: str, defect_entry: DefectEntry) -> dict:
    """
    Generates a dictionary of key information about the DefectEntry
    Args:
        defect_species (str): Name of the defect.
        defect_entry (DefectEntry):
            Pymatgen DefectEntry to generate transformation information for.
    """
    defect_supercell = defect_entry.sc_entry.structure
    defect_site = _get_defect_site(defect_entry)
    bulk_defect_site = _get_bulk_defect_site(defect_entry)

    defect_type = defect_entry.defect.defect_type.name.lower()
    if defect_type in ["antisite", "substitution"]:  # get substitution_specie
        substitution_specie = str(
            defect_entry.defect.site.specie.symbol
        )  # Current species occupying the defect site (e.g. the substitution)
    else:
        substitution_specie = None

    frac_coords = defect_site.frac_coords
    approx_coords = f"~[{frac_coords[0]:.4f},{frac_coords[1]:.4f},{frac_coords[2]:.4f}]"
    # Note this gets truncated to 40 characters in the CONTCAR: (this should be less than
    # 40 chars in all cases):
    poscar_comment = f"{defect_species} {approx_coords}"

    transformation_dict = {
        "defect_type": defect_entry.defect.defect_type,
        "defect_unit_cell_site": defect_entry.defect.site,
        "defect_supercell_site": defect_site,
        "bulk_supercell_site": bulk_defect_site,
        "unit_cell_multiplicity": defect_entry.defect.multiplicity,
        "charge": f"{defect_entry.charge_state:+d}",
        "substitution_specie": substitution_specie,
        "POSCAR Comment": poscar_comment,
        "Defect": defect_entry.defect,
        "DefectEntry": defect_entry,
    }

    return transformation_dict


def _prepare_vasp_files(
    defect_species: str,
    single_defect_entry: DefectEntry,
    defect_dir: str = ".",
    subfolder: str = "vasp_gam",
    vasp_type: str = "gam",
    user_incar_settings: Optional[dict] = None,
    user_kpoints_settings: Optional[Union[dict, Kpoints]] = None,
    user_potcar_functional: str = "PBE_54",
    user_potcar_settings: Optional[dict] = None,
    unperturbed_poscar: bool = False,
    write_files: bool = True,
    write_transformation_info: bool = True,
) -> DefectRelaxSet:
    """
    Prepare DefectRelaxSet object for VASP defect supercell input file generation

    Args:
        defect_species (str): Name of the defect species.
        single_defect_entry (DefectEntry):
            Pymatgen DefectEntry to prepare VASP input files for.
        defect_dir (str):
            Folder in which to create the VASP defect calculation inputs.
            Default is the current directory.
        subfolder (str):
            Output folder structure is `<defect_dir>/<subfolder>` where
            `subfolder` = 'vasp_gam' by default. Setting `subfolder` to `None` will write
            the `vasp_gam` input files directly to the `<defect_dir>` folder,
            with no subfolders created.
        vasp_type (str): Which vasp_xxx binary to use (i.e. "gam", "std" or "ncl).
            Default is "gam".
        user_incar_settings (dict):
            Dictionary of user INCAR settings (AEXX, NCORE etc.) to override default settings.
            Highly recommended to look at output INCARs or the `HSE06_RelaxSet.yaml` and
            `DefectSet.yaml` files in the `doped` folder, to see what the default INCAR settings
            are. Note that any flags that aren't numbers or True/False need to be input as
            strings with quotation marks (e.g. `{"ALGO": "All"}`).
            (default: None)
        user_kpoints_settings (dict or Kpoints):
            Dictionary of user KPOINTS settings (in pymatgen DictSet() format) e.g.,
            {"reciprocal_density": 1000}, or a Kpoints object. Default KPOINTS is Gamma-point-only
            if `vasp_type = 'vasp_gam'`, else a Gamma-centred 2 x 2 x 2 mesh.
        user_potcar_functional (str): POTCAR functional to use (default = "PBE_54")
        user_potcar_settings (dict): Override the default POTCARs, e.g. {"Li": "Li_sv"}. See
            `doped/PotcarSet.yaml` for the default `POTCAR` set.
        unperturbed_poscar (bool):
            If True, write the unperturbed defect POSCARs to the generated folders as well. Not
            recommended, as the recommended workflow is to initially perform vasp_gam
            ground-state structure searching using ShakeNBreak (see example notebook;
            https://shakenbreak.readthedocs.io), then continue the vasp_std relaxations from the
            'Groundstate' CONTCARs.
            (default: False)
        write_files (bool): Whether to write the VASP input files to disk. Default is True.
        write_transformation_info (bool): Whether to write the transformation.json file (with info
            about generated defect supercell, mainly for logging purposes). Default is False.

    Returns:
        `DefectRelaxSet` object for the supplied defect entry.
    """
    defect_supercell = single_defect_entry.sc_entry.structure
    transformation_dict = _get_transformation_info(defect_species, single_defect_entry)

    if subfolder is None:
        vaspinputdir = defect_dir  # defect_dir is = {output_dir}/{defect_species}
    else:
        vaspinputdir = f"{defect_dir}/{subfolder}"

    if user_potcar_functional is not None:
        potcars = any(["VASP_PSP_DIR" in i for i in SETTINGS])
        if not potcars:
            if vasp_type == "gam":
                warnings.warn(
                    "POTCAR directory not set up with pymatgen (see the doped homepage: "
                    "https://github.com/SMTG-UCL/doped for instructions on setting this up). "
                    "This is required to generate `POTCAR` and `INCAR` files (to set `NELECT` "
                    "and `NUPDOWN`), so only `POSCAR` files will be generated."
                )
            elif unperturbed_poscar:  # vasp_std, vasp_ncl
                warnings.warn(
                    "POTCAR directory not set up with pymatgen, so only unperturbed POSCAR files "
                    "will be generated (POTCARs also needed to determine appropriate NELECT "
                    "setting in INCAR files)"
                )
            else:
                raise ValueError(
                    "POTCAR directory not set up with pymatgen (see the doped homepage: "
                    "https://github.com/SMTG-UCL/doped for instructions on setting this up). "
                    "This is required to generate `POTCAR` and `INCAR` files (to set `NELECT` "
                    "and `NUPDOWN`), so no input files will be generated here."
                )
            return None

    relax_set = deepcopy(default_relax_set)
    potcar_dict = deepcopy(default_potcar_dict)
    if user_potcar_functional:
        potcar_dict["POTCAR_FUNCTIONAL"] = user_potcar_functional
    if user_potcar_settings:
        potcar_dict["POTCAR"].update(user_potcar_settings)
    relax_set.update(potcar_dict)

    if user_incar_settings:
        for k in user_incar_settings.keys():
            # check INCAR flags and warn if they don't exist (typos)
            if k not in incar_params.keys():
                # this code is taken from pymatgen.io.vasp.inputs
                warnings.warn(  # but only checking keys, not values so we can add comments etc
                    f"Cannot find {k} from your user_incar_settings in the list of INCAR flags",
                    BadIncarWarning,
                )
        relax_set["INCAR"].update(user_incar_settings)
        if "EDIFF" not in user_incar_settings:  # set scaled
            relax_set["INCAR"]["EDIFF"] = scaled_ediff(len(defect_supercell))
        if (
            "EDIFFG" not in user_incar_settings
            and vasp_type == "ncl"
            and relax_set["INCAR"]["IBRION"] == -1
            and relax_set["INCAR"]["NSW"] == 0
        ):
            # default SOC = singlepoint calc, remove EDIFFG & POTIM from INCAR to avoid confusion:
            del relax_set["INCAR"]["EDIFFG"]
            del relax_set["INCAR"]["POTIM"]

    if user_kpoints_settings is None and vasp_type == "gam":
        user_kpoints_settings = Kpoints().from_dict(
            {
                "comment": "kpoints from doped.vasp_gam_files()",
                "generation_style": "Gamma",
            }
        )
    elif user_kpoints_settings is None:  # vasp_std, vasp_ncl
        # use default vasp_std/ncl KPOINTS, Gamma-centred 2 x 2 x 2 mesh
        user_kpoints_settings = Kpoints.from_dict(
            {
                "comment": "kpoints from doped",
                "generation_style": "Gamma",
                "kpoints": [[2, 2, 2]],
            }
        )

    # set POSCAR comment:
    poscar_comment = transformation_dict.get("POSCAR Comment")

    defect_relax_set = DefectRelaxSet(
        defect_supercell,
        config_dict=relax_set,
        user_kpoints_settings=user_kpoints_settings,  # accepts Kpoints obj, so we can set comment
        charge=single_defect_entry.charge_state,
        poscar_comment=poscar_comment,
    )
    defect_relax_set.output_dir = (
        vaspinputdir  # assign attribute to later use in write_input()
    )

    if write_files:
        if not os.path.exists(defect_relax_set.output_dir):
            os.makedirs(defect_relax_set.output_dir)

        if unperturbed_poscar:
            defect_relax_set.poscar.write_file(defect_relax_set.output_dir + "/POSCAR")

        if write_transformation_info:  # write transformation.json file
            dumpfn(
                transformation_dict,
                f"{defect_relax_set.output_dir}/transformation.json",
            )

    return defect_relax_set


def _format_defect_entries_input(
    defect_entries: Union[
        DefectsGenerator, Dict[str, DefectEntry], List[DefectEntry], DefectEntry
    ]
) -> Dict[str, DefectEntry]:
    """
    Helper function to format input `defect_entries` into a dictionary of `DefectEntry` objects,
    to use in `vasp_xxx_files()` functions.

    Args:
        defect_entries (Union[DefectsGenerator, Dict[DefectEntry], List[DefectEntry], DefectEntry]):
            Either a `DefectsGenerator` object, or a dictionary of, list of or single `DefectEntry`
            object, for which to generate VASP input files. If a `DefectsGenerator` object or a
            dictionary (-> {defect_species: DefectEntry}), the defect folder names will be set equal
            to `defect_species`. If a list or single `DefectEntry` object is provided, the defect folder
            names will be set equal to `DefectEntry.name` if the `name` attribute is set, otherwise
            generated according to the `doped` convention.
    """
    if isinstance(defect_entries, DefectsGenerator):
        defect_entries = defect_entries.defect_entries
    elif isinstance(defect_entries, DefectEntry):
        defect_entries = [defect_entries]
    if isinstance(
        defect_entries, list
    ):  # also catches case where defect_entries is a single DefectEntry, from above
        # convert to dict with doped names as keys:
        defect_naming_dict = {}
        defect_entry_dict = {}
        for defect_entry in defect_entries:
            if not hasattr(defect_entry, "name"):
                defect_name_wout_charge = _get_defect_name_from_obj(defect_entry.defect)
                defect_name_wout_charge = _update_defect_dict(
                    defect_entry, defect_name_wout_charge, defect_naming_dict
                )
                defect_name = (
                    defect_name_wout_charge
                    + f"_{'+' if defect_entry.charge_state > 0 else ''}{defect_entry.charge_state}"
                )
                defect_entry.name = defect_name  # set name attribute

            defect_entry_dict[defect_entry.name] = defect_entry
        defect_entries = defect_entry_dict

    if isinstance(defect_entries, dict):  # check correct format
        if not all(
            isinstance(defect_entry, DefectEntry)
            for defect_entry in defect_entries.values()
        ):
            raise TypeError(
                f"Input defect_entries dict must be of the form {{defect_name: DefectEntry}}, got dict "
                f"with values of type {[type(value) for value in defect_entries.values()]} instead"
            )

    return defect_entries


# TODO: Implement renaming folders like SnB if we try to write a folder that already exists, and the structures don't
#  match (otherwise overwrite)
def vasp_gam_files(
    defect_entries: Union[
        DefectsGenerator, Dict[str, DefectEntry], List[DefectEntry], DefectEntry
    ],
    output_dir: str = ".",
    subfolder: str = "vasp_gam",
    user_incar_settings: Optional[dict] = None,
    user_potcar_functional: str = "PBE_54",
    user_potcar_settings: Optional[dict] = None,
    write_files: bool = True,
    write_transformation_info: bool = False,
    **kwargs,  # to allow POTCAR testing on GH Actions
) -> Dict[str, DefectRelaxSet]:
    """
    Generates input files for VASP Gamma-point-only (`vasp_gam`) coarse defect supercell
    relaxations. See the `HSE06_RelaxSet.yaml` and `DefectSet.yaml` files in the `doped` folder
    for the default `INCAR` settings, and `PotcarSet.yaml` for the default `POTCAR` settings.

    Note that any changes to the default `INCAR`/`POTCAR` settings should be consistent with
    those used for competing phase (chemical potential) calculations.

    Args:
        defect_entries (Union[DefectsGenerator, Dict[DefectEntry], List[DefectEntry], DefectEntry]):
            Either a `DefectsGenerator` object, or a dictionary of, list of or single `DefectEntry`
            object, for which to generate VASP input files. If a `DefectsGenerator` object or a
            dictionary (-> {defect_species: DefectEntry}), the defect folder names will be set equal
            to `defect_species`. If a list or single `DefectEntry` object is provided, the defect folder
            names will be set equal to `DefectEntry.name` if the `name` attribute is set, otherwise
            generated according to the `doped` convention.
        output_dir (str):
            Folder in which to create the VASP defect calculation inputs folders. Default is the
            current directory.
        subfolder (str):
            Output folder structure is `<output_dir>/<defect_species>/<subfolder>` where `subfolder` =
            'vasp_gam' by default. Setting `subfolder` to `None` will write the `vasp_gam` input
            files directly to the `<defect_species>` folder, with no subfolders created.
        user_incar_settings (dict):
            Dictionary of user INCAR settings (AEXX, NCORE etc.) to override default settings.
            Highly recommended to look at output INCARs or the `HSE06_RelaxSet.yaml` and
            `DefectSet.yaml` files in the `doped` folder, to see what the default INCAR settings
            are. Note that any flags that aren't numbers or True/False need to be input as
            strings with quotation marks (e.g. `{"ALGO": "All"}`).
            (default: None)
        user_potcar_functional (str): POTCAR functional to use (default = "PBE_54")
        user_potcar_settings (dict): Override the default POTCARs, e.g. {"Li": "Li_sv"}. See
            `doped/PotcarSet.yaml` for the default `POTCAR` set.
        write_files (bool): Whether to write the VASP input files to disk. Default is True.
        write_transformation_info (bool): Whether to write the transformation.json file (with info
            about generated defect supercell, mainly for logging purposes). Default is False.

    Returns:
        Dictionary of {defect_species: `DefectRelaxSet`} for each defect species in the input
        defect_dict. `DefectRelaxSet` is a subclass of `pymatgen`'s `DictSet` class,
        with `incar`, `poscar`, `kpoints` and `potcar` attributes, containing information on the
        generated files.
    """
    # check type of input `defect_entries`, and format if necessary:
    defect_entries = _format_defect_entries_input(defect_entries)

    defect_relax_set_dict = {}
    for defect_species, single_defect_entry in defect_entries.items():
        defect_relax_set = _prepare_vasp_files(
            defect_species=defect_species,
            single_defect_entry=single_defect_entry,
            defect_dir=f"{output_dir}/{defect_species}",
            subfolder=subfolder,
            vasp_type="gam",
            user_incar_settings=user_incar_settings,
            user_kpoints_settings=None,  # defaults to Gamma-only when vasp_type="gam"
            user_potcar_functional=user_potcar_functional,
            user_potcar_settings=user_potcar_settings,
            unperturbed_poscar=True,  # write POSCAR for vasp_gam_files()
            write_files=write_files,
            write_transformation_info=write_transformation_info,
        )

        if write_files:
            defect_relax_set.write_input(
                defect_relax_set.output_dir,
                **kwargs,  # kwargs to allow POTCAR testing on GH Actions
            )

        defect_relax_set_dict[defect_species] = defect_relax_set

    return defect_relax_set_dict


def vasp_std_files(
    defect_entries: Union[
        DefectsGenerator, Dict[str, DefectEntry], List[DefectEntry], DefectEntry
    ],
    output_dir: str = ".",
    subfolder: str = "vasp_std",
    user_incar_settings: Optional[dict] = None,
    user_kpoints_settings: Optional[Union[dict, Kpoints]] = None,
    user_potcar_functional: str = "PBE_54",
    user_potcar_settings: Optional[dict] = None,
    unperturbed_poscar: bool = False,
    write_files: bool = True,
    write_transformation_info: bool = False,
) -> Dict[str, DefectRelaxSet]:
    """
    Generates INCAR, POTCAR and KPOINTS for `vasp_std` defect supercell relaxations. By default
    does not generate POSCAR (input structure) files, as these should be taken from `ShakeNBreak`
    calculations (via `snb-groundstate`) or `vasp_gam` calculations (using
    `vasp_input.vasp_gam_files()`, and `cp vasp_gam/CONTCAR vasp_std/POSCAR`). See the
    `HSE06_RelaxSet.yaml` and `DefectSet.yaml` files in the `doped` folder for the default
    `INCAR` settings, and `PotcarSet.yaml` for the default `POTCAR` settings.

    Note that any changes to the default `INCAR`/`POTCAR` settings should be consistent with
    those used for competing phase (chemical potential) calculations.

    Args:
        defect_entries (Union[DefectsGenerator, Dict[DefectEntry], List[DefectEntry], DefectEntry]):
            Either a `DefectsGenerator` object, or a dictionary of, list of or single `DefectEntry`
            object, for which to generate VASP input files. If a `DefectsGenerator` object or a
            dictionary (-> {defect_species: DefectEntry}), the defect folder names will be set equal
            to `defect_species`. If a list or single `DefectEntry` object is provided, the defect folder
            names will be set equal to `DefectEntry.name` if the `name` attribute is set, otherwise
            generated according to the `doped` convention.
        output_dir (str):
            Folder in which to create the VASP defect calculation inputs folders. Default is the
            current directory.
        subfolder (str):
            Output folder structure is `<output_dir>/<defect_species>/<subfolder>` where `subfolder` =
            'vasp_std' by default. Setting `subfolder` to `None` will write the vasp_std input
            files directly to the `<defect_species>` folder, with no subfolders created.
        user_incar_settings (dict):
            Dictionary of user INCAR settings (AEXX, NCORE etc.) to override default settings.
            Highly recommended to look at output INCARs or doped.vasp_input
            source code, to see what the default INCAR settings are. Note that any flags that
            aren't numbers or True/False need to be input as strings with quotation marks
            (e.g. `{"ALGO": "All"}`).
            (default: None)
        user_kpoints_settings (dict or Kpoints):
            Dictionary of user KPOINTS settings (in pymatgen DictSet() format) e.g.,
            {"reciprocal_density": 1000}, or a Kpoints object. Default KPOINTS is a Gamma-centred
            2 x 2 x 2 mesh.
            (default: None)
        user_potcar_functional (str): POTCAR functional to use (default = "PBE_54")
        user_potcar_settings (dict): Override the default POTCARs, e.g. {"Li": "Li_sv"}. See
            `doped/PotcarSet.yaml` for the default `POTCAR` set.
        unperturbed_poscar (bool):
            If True, write the unperturbed defect POSCARs to the generated folders as well. Not
            recommended, as the recommended workflow is to initially perform vasp_gam
            ground-state structure searching using ShakeNBreak (see example notebook;
            https://shakenbreak.readthedocs.io), then continue the vasp_std relaxations from the
            'Groundstate' CONTCARs.
            (default: False)
        write_files (bool): Whether to write the VASP input files to disk. Default is True.
        write_transformation_info (bool): Whether to write the transformation.json file (with info
            about generated defect supercell, mainly for logging purposes). Default is False.

    Returns:
        Dictionary of {defect_species: `DefectRelaxSet`} for each defect species in the input
        defect_dict. `DefectRelaxSet` is a subclass of `pymatgen`'s `DictSet` class,
        with `incar`, `poscar` (unperturbed), `kpoints` and `potcar` attributes, containing
        information on the generated files.
    """
    # check type of input `defect_entries`, and format if necessary:
    defect_entries = _format_defect_entries_input(defect_entries)

    vaspstdincardict = {
        "KPAR": 2,  # vasp_std calculations so multiple k-points, likely quicker with this
    }
    if user_incar_settings is not None:
        vaspstdincardict.update(user_incar_settings)

    defect_relax_set_dict = {}
    for defect_species, single_defect_entry in defect_entries.items():
        defect_relax_set = _prepare_vasp_files(
            defect_species=defect_species,
            single_defect_entry=single_defect_entry,
            defect_dir=f"{output_dir}/{defect_species}",
            subfolder=subfolder,
            vasp_type="std",
            user_incar_settings=vaspstdincardict,
            user_kpoints_settings=user_kpoints_settings,
            user_potcar_functional=user_potcar_functional,
            user_potcar_settings=user_potcar_settings,
            unperturbed_poscar=unperturbed_poscar,
            write_files=write_files,
            write_transformation_info=write_transformation_info,
        )

        if write_files:  # write INCAR, KPOINTS and POTCAR files
            # then use `write_file()`s rather than `write_input()` to avoid writing POSCARs
            defect_relax_set.incar.write_file(defect_relax_set.output_dir + "/INCAR")
            defect_relax_set.kpoints.write_file(
                defect_relax_set.output_dir + "/KPOINTS"
            )
            if user_potcar_functional is not None:  # for GH Actions testing
                defect_relax_set.potcar.write_file(
                    defect_relax_set.output_dir + "/POTCAR"
                )

        defect_relax_set_dict[defect_species] = defect_relax_set

    return defect_relax_set_dict


def vasp_ncl_files(
    defect_entries: Union[
        DefectsGenerator, Dict[str, DefectEntry], List[DefectEntry], DefectEntry
    ],
    output_dir: str = ".",
    subfolder: str = "vasp_ncl",
    user_incar_settings: Optional[dict] = None,
    user_kpoints_settings: Optional[Union[dict, Kpoints]] = None,
    user_potcar_functional: str = "PBE_54",
    user_potcar_settings: Optional[dict] = None,
    unperturbed_poscar: bool = False,
    write_files: bool = True,
    write_transformation_info: bool = False,
) -> Dict[str, DefectRelaxSet]:
    """
    Generates INCAR, POTCAR and KPOINTS for `vasp_ncl` (i.e. spin-orbit coupling (SOC)) defect
    supercell singlepoint calculations. By default does not generate POSCAR (input structure)
    files, as these should be taken from `vasp_std` relaxations (i.e.
    `cp vasp_std/CONTCAR vasp_ncl/POSCAR`), which themselves likely originated from `ShakeNBreak`
    calculations (via `snb-groundstate`). See the `HSE06_RelaxSet.yaml` and `DefectSet.yaml`
    files in the `doped` folder for the default `INCAR` settings, and `PotcarSet.yaml` for the
    default `POTCAR` settings.

    Note that any changes to the default `INCAR`/`POTCAR` settings should be consistent with
    those used for competing phase (chemical potential) calculations.

    Args:
        defect_entries (Union[DefectsGenerator, Dict[DefectEntry], List[DefectEntry], DefectEntry]):
            Either a `DefectsGenerator` object, or a dictionary of, list of or single `DefectEntry`
            object, for which to generate VASP input files. If a `DefectsGenerator` object or a
            dictionary (-> {defect_species: DefectEntry}), the defect folder names will be set equal
            to `defect_species`. If a list or single `DefectEntry` object is provided, the defect folder
            names will be set equal to `DefectEntry.name` if the `name` attribute is set, otherwise
            generated according to the `doped` convention.
        output_dir (str):
            Folder in which to create the VASP defect calculation inputs folders. Default is the
            current directory.
        subfolder (str):
            Output folder structure is `<output_dir>/<defect_species>/<subfolder>` where `subfolder` =
            'vasp_ncl' by default. Setting `subfolder` to `None` will write the vasp_ncl input
            files directly to the `<defect_species>` folder, with no subfolders created.
        user_incar_settings (dict):
            Dictionary of user INCAR settings (AEXX, NCORE etc.) to override default settings.
            Highly recommended to look at output INCARs or doped.vasp_input
            source code, to see what the default INCAR settings are. Note that any flags that
            aren't numbers or True/False need to be input as strings with quotation marks
            (e.g. `{"ALGO": "All"}`).
            (default: None)
        user_kpoints_settings (dict or Kpoints):
            Dictionary of user KPOINTS settings (in pymatgen DictSet() format) e.g.,
            {"reciprocal_density": 1000}, or a Kpoints object. Default KPOINTS is a Gamma-centred
            2 x 2 x 2 mesh.
            (default: None)
        user_potcar_functional (str): POTCAR functional to use (default = "PBE_54")
        user_potcar_settings (dict): Override the default POTCARs, e.g. {"Li": "Li_sv"}. See
            `doped/PotcarSet.yaml` for the default `POTCAR` set.
        unperturbed_poscar (bool):
            If True, write the unperturbed defect POSCARs to the generated folders as well. Not
            recommended, as the recommended workflow is to initially perform vasp_gam
            ground-state structure searching using ShakeNBreak (see example notebook;
            https://shakenbreak.readthedocs.io), then continue the vasp_std relaxations from the
            'Groundstate' CONTCARs, before doing final vasp_ncl singleshot calculations if SOC is
            important.
            (default: False)
        write_files (bool): Whether to write the VASP input files to disk. Default is True.
        write_transformation_info (bool): Whether to write the transformation.json file (with info
            about generated defect supercell, mainly for logging purposes). Default is False.

    Returns:
        Dictionary of {defect_species: `DefectRelaxSet`} for each defect species in the input
        defect_dict. `DefectRelaxSet` is a subclass of `pymatgen`'s `DictSet` class,
        with `incar`, `poscar` (unperturbed), `kpoints` and `potcar` attributes, containing
        information on the generated files.
    """
    # check type of input `defect_entries`, and format if necessary:
    defect_entries = _format_defect_entries_input(defect_entries)

    vaspnclincardict = {
        "EDIFF": 1e-06,  # tight EDIFF for final energy and converged DOS",
        "KPAR": 2,  # vasp_ncl calculations so likely multiple k-points, likely quicker with this
        "LSORBIT": True,
        "NSW": 0,  # no ionic relaxation"
        "IBRION": -1,  # no ionic relaxation"
    }
    if user_incar_settings is not None:
        vaspnclincardict.update(user_incar_settings)

    defect_relax_set_dict = {}
    for defect_species, single_defect_entry in defect_entries.items():
        defect_relax_set = _prepare_vasp_files(
            defect_species=defect_species,
            single_defect_entry=single_defect_entry,
            defect_dir=f"{output_dir}/{defect_species}",
            subfolder=subfolder,
            vasp_type="ncl",
            user_incar_settings=vaspnclincardict,
            user_kpoints_settings=user_kpoints_settings,
            user_potcar_functional=user_potcar_functional,
            user_potcar_settings=user_potcar_settings,
            unperturbed_poscar=unperturbed_poscar,
            write_files=write_files,
            write_transformation_info=write_transformation_info,
        )

        if write_files:
            # then use `write_file()`s rather than `write_input()` to avoid writing POSCARs
            defect_relax_set.incar.write_file(defect_relax_set.output_dir + "/INCAR")
            defect_relax_set.kpoints.write_file(
                defect_relax_set.output_dir + "/KPOINTS"
            )
            if user_potcar_functional is not None:  # for GH Actions testing
                defect_relax_set.potcar.write_file(
                    defect_relax_set.output_dir + "/POTCAR"
                )

        defect_relax_set_dict[defect_species] = defect_relax_set

    return defect_relax_set_dict


# TODO: Remove these functions once confirming all functionality is in `chemical_potentials.py`;
# need `vasp_ncl_chempot` generation, `vaspup2.0` `input` folder with `CONFIG` generation as an
# option, improve chemical_potentials docstrings (i.e. mention defaults, note in notebooks if changing
# `INCAR`/`POTCAR` settings for competing phase production calcs, should also do with defect
# supercell calcs (and note this in vasp_input as well)), ensure consistent INCAR tags in defect
# supercell defaults and competing phase defaults, point to DefectSet in docstrings for defaults
# (noting the other INCAR tags that are changed).
def _vasp_converge_files(
    structure: "pymatgen.core.Structure",
    input_dir: Optional[str] = None,
    incar_settings: Optional[dict] = None,
    potcar_settings: Optional[dict] = None,
    config: Optional[str] = None,
) -> None:
    """
    Generates input files for single-shot GGA convergence test calculations.
    Automatically sets ISMEAR (in INCAR) to 2 (if metallic) or 0 if not.
    Recommended to use with vaspup2.0
    Args:
        structure (Structure object):
            Structure to create input files for.
        input_dir (str):
            Folder in which to create 'input' folder with VASP input files.
            (default: None)
        incar_settings (dict):
            Dictionary of user INCAR settings (AEXX, NCORE etc.) to override default settings.
            Highly recommended to look at output INCARs or doped.vasp_input
            source code, to see what the default INCAR settings are. Note that any flags that
            aren't numbers or True/False need to be input as strings with quotation marks
            (e.g. `{"ALGO": "All"}`).
            (default: None)
        config (str):
            CONFIG file string. If provided, will also write the CONFIG file (to automate
            convergence tests with vaspup2.0) to each 'input' directory.
            (default: None)
        potcar_settings (dict):
            Dictionary of user POTCAR settings to override default settings.
            Highly recommended to look at `default_potcar_dict` from doped.vasp_input to see what
            the (Pymatgen) syntax and doped default settings are.
            (default: None)
    """

    # Variable parameters first
    vaspconvergeincardict = {
        "# May need to change ISMEAR, NCORE, KPAR, AEXX, ENCUT, NUPDOWN, "
        + "ISPIN": "variable parameters",
        "NUPDOWN": "0 # But could be >0 if magnetic behaviour present",
        "NCORE": 12,
        "#KPAR": 1,
        "ENCUT": 400,
        "ISMEAR": "0 # Non-metal, use Gaussian smearing",
        "ISPIN": "1 # Change to 2 if spin polarisation or magnetic behaviour present",
        "GGA": "PS",  # PBEsol
        "ALGO": "Normal # Change to All if ZHEGV, FEXCP/F or ZBRENT errors encountered",
        "EDIFF": 1e-06,
        "EDIFFG": -0.01,
        "IBRION": -1,
        "ISIF": 3,
        "LASPH": True,
        "LORBIT": 14,
        "LREAL": False,
        "LWAVE": "False # Save filespace, shouldn't need WAVECAR from convergence tests",
        "NEDOS": 2000,
        "NELM": 100,
        "NSW": 0,
        "PREC": "Accurate",
        "SIGMA": 0.2,
    }
    if all(is_metal(element) for element in structure.composition.elements):
        vaspconvergeincardict[
            "ISMEAR"
        ] = "2 # Metal, use Methfessel-Paxton smearing scheme"
    if incar_settings:
        for (
            k
        ) in incar_settings.keys():  # check INCAR flags and warn if they don't exist (
            # typos)
            if (
                k not in incar_params.keys()
            ):  # this code is taken from pymatgen.io.vasp.inputs
                warnings.warn(  # but only checking keys, not values so we can add comments etc
                    "Cannot find %s from your user_incar_settings in the list of INCAR flags"
                    % (k),
                    BadIncarWarning,
                )
        vaspconvergeincardict.update(incar_settings)

    # Directory
    vaspconvergeinputdir = input_dir + "/input/" if input_dir else "VASP_Files/input/"
    if not os.path.exists(vaspconvergeinputdir):
        os.makedirs(vaspconvergeinputdir)

    # POTCAR
    potcar_dict = deepcopy(default_potcar_dict)
    if potcar_settings:
        if "POTCAR_FUNCTIONAL" in potcar_settings.keys():
            potcar_dict["POTCAR_FUNCTIONAL"] = potcar_settings["POTCAR_FUNCTIONAL"]
        if "POTCAR" in potcar_settings.keys():
            potcar_dict["POTCAR"].update(potcar_settings.pop("POTCAR"))
    vaspconvergeinput = DictSet(structure, config_dict=potcar_dict)
    vaspconvergeinput.potcar.write_file(vaspconvergeinputdir + "POTCAR")

    vaspconvergekpts = Kpoints().from_dict(
        {"comment": "Kpoints from vasp_gam_files", "generation_style": "Gamma"}
    )
    vaspconvergeincar = Incar.from_dict(vaspconvergeincardict)
    vaspconvergeincar.write_file(vaspconvergeinputdir + "INCAR")

    vaspconvergeposcar = Poscar(structure)
    vaspconvergeposcar.write_file(vaspconvergeinputdir + "POSCAR")

    vaspconvergekpts.write_file(vaspconvergeinputdir + "KPOINTS")
    # generate CONFIG file
    if config:
        with open(vaspconvergeinputdir + "CONFIG", "w+") as config_file:
            config_file.write(config)
        with open(vaspconvergeinputdir + "CONFIG", "a") as config_file:
            config_file.write(f"""\nname="{input_dir[13:]}" # input_dir""")


# Input files for vasp_std


def _vasp_std_chempot(
    structure: "pymatgen.core.Structure",
    input_dir: Optional[str] = None,
    incar_settings: Optional[dict] = None,
    kpoints_settings: Optional[dict] = None,
    potcar_settings: Optional[dict] = None,
) -> None:
    """
    Generates POSCAR, INCAR, POTCAR and KPOINTS for vasp_std chemical potentials relaxation.:
    Args:
        structure (Structure object):
            Structure to create input files for.
        input_dir (str):
            Folder in which to create vasp_std calculation inputs folder
            (default: None)
        incar_settings (dict):
            Dictionary of user INCAR settings (AEXX, NCORE etc.) to override default settings.
            Highly recommended to look at output INCARs or doped.vasp_input
            source code, to see what the default INCAR settings are. Note that any flags that
            aren't numbers or True/False need to be input as strings with quotation marks
            (e.g. `{"ALGO": "All"}`).
            (default: None)
        kpoints_settings (dict):
            Dictionary of user KPOINTS settings (in pymatgen Kpoints.from_dict() format). Common
            options would be "generation_style": "Monkhorst" (rather than "Gamma"),
            and/or "kpoints": [[3, 3, 1]] etc.
            Default KPOINTS is Gamma-centred 2 x 2 x 2 mesh.
            (default: None)
        potcar_settings (dict):
            Dictionary of user POTCAR settings to override default settings.
            Highly recommended to look at `default_potcar_dict` from doped.vasp_input to see what
            the (Pymatgen) syntax and doped default settings are.
            (default: None)
    """
    # INCAR Parameters
    vaspstdincardict = {
        "# May need to change NCORE, KPAR, ENCUT"
        + "ISPIN, POTIM": "variable parameters",
        "NCORE": 12,
        "KPAR": 2,
        "AEXX": 0.25,
        "ENCUT": 400,
        "POTIM": 0.2,
        "LSUBROT": "False # Change to True if relaxation poorly convergent",
        "ICORELEVEL": "0 # Needed if using the Kumagai-Oba (eFNV) anisotropic charge correction",
        "ALGO": "Normal # Change to All if ZHEGV, FEXCP/F or ZBRENT errors encountered",
        "EDIFF": 1e-06,  # May need to reduce for tricky relaxations",
        "EDIFFG": -0.01,
        "HFSCREEN": 0.2,  # assuming HSE06
        "IBRION": "1 # May need to change to 2 for difficult/poorly-convergent relaxations",
        "ISIF": 3,
        "ISMEAR": 0,
        "LASPH": True,
        "LHFCALC": True,
        "LORBIT": 14,
        "LREAL": False,
        "LVHAR": "True # Needed if using the Freysoldt (FNV) charge correction scheme",
        "LWAVE": True,
        "NEDOS": 2000,
        "NELM": 100,
        "NSW": 200,
        "PREC": "Accurate",
        "PRECFOCK": "Fast",
        "SIGMA": 0.05,
    }

    # Directory
    vaspstdinputdir = input_dir + "/vasp_std/" if input_dir else "VASP_Files/vasp_std/"
    if not os.path.exists(vaspstdinputdir):
        os.makedirs(vaspstdinputdir)

    # POTCAR
    potcar_dict = default_potcar_dict
    if potcar_settings:
        if "POTCAR_FUNCTIONAL" in potcar_settings.keys():
            potcar_dict["POTCAR_FUNCTIONAL"] = potcar_settings["POTCAR_FUNCTIONAL"]
        if "POTCAR" in potcar_settings.keys():
            potcar_dict["POTCAR"].update(potcar_settings.pop("POTCAR"))
    vaspstdinput = DictSet(structure, config_dict=potcar_dict)
    vaspstdinput.potcar.write_file(vaspstdinputdir + "POTCAR")

    if all(is_metal(element) for element in structure.composition.elements):
        vaspstdincardict["ISMEAR"] = "2 # Metal, use Methfessel-Paxton smearing scheme"
    if all(is_metal(element) for element in structure.composition.elements):
        vaspstdincardict["SIGMA"] = 0.02

    if incar_settings:
        for (
            k
        ) in (
            incar_settings.keys()
        ):  # check INCAR flags and warn if they don't exist (typos)
            if (
                k not in incar_params.keys()
            ):  # this code is taken from pymatgen.io.vasp.inputs
                warnings.warn(  # but only checking keys, not values so we can add comments etc
                    "Cannot find %s from your user_incar_settings in the list of INCAR flags"
                    % (k),
                    BadIncarWarning,
                )
        vaspstdincardict.update(incar_settings)

    # POSCAR
    vaspstdposcar = Poscar(structure)
    vaspstdposcar.write_file(vaspstdinputdir + "POSCAR")

    # KPOINTS
    vaspstdkpointsdict = {
        "comment": "Kpoints from doped.vasp_std_files",
        "generation_style": "Gamma",  # Set to Monkhorst for Monkhorst-Pack generation
        "kpoints": [[2, 2, 2]],
    }
    if kpoints_settings:
        vaspstdkpointsdict.update(kpoints_settings)
    vaspstdkpts = Kpoints.from_dict(vaspstdkpointsdict)
    vaspstdkpts.write_file(vaspstdinputdir + "KPOINTS")

    # INCAR
    vaspstdincar = Incar.from_dict(vaspstdincardict)
    with zopen(vaspstdinputdir + "INCAR", "wt") as incar_file:
        incar_file.write(vaspstdincar.get_string())


# Input files for vasp_ncl


def _vasp_ncl_chempot(
    structure: "pymatgen.core.Structure",
    input_dir: Optional[str] = None,
    incar_settings: Optional[dict] = None,
    kpoints_settings: Optional[dict] = None,
    potcar_settings: Optional[dict] = None,
) -> None:
    """
    Generates INCAR, POTCAR and KPOINTS for vasp_ncl chemical potentials relaxation.
    Take CONTCAR from vasp_std for POSCAR.:
    Args:
        structure (Structure object):
            Structure to create input files for.
        input_dir (str):
            Folder in which to create vasp_ncl calculation inputs folder
            (default: None)
        incar_settings (dict):
            Dictionary of user INCAR settings (AEXX, NCORE etc.) to override default settings.
            Highly recommended to look at output INCARs or doped.vasp_input
            source code, to see what the default INCAR settings are. Note that any flags that
            aren't numbers or True/False need to be input as strings with quotation marks
            (e.g. `{"ALGO": "All"}`).
            (default: None)
        kpoints_settings (dict):
            Dictionary of user KPOINTS settings (in pymatgen Kpoints.from_dict() format). Common
            options would be "generation_style": "Monkhorst" (rather than "Gamma"),
            and/or "kpoints": [[3, 3, 1]] etc.
            Default KPOINTS is Gamma-centred 2 x 2 x 2 mesh.
            (default: None)
        potcar_settings (dict):
            Dictionary of user POTCAR settings to override default settings.
            Highly recommended to look at `default_potcar_dict` from doped.vasp_input to see what
            the (Pymatgen) syntax and doped default settings are.
            (default: None)
    """
    # INCAR Parameters
    vaspnclincardict = {
        "# May need to change NELECT, NCORE, KPAR, AEXX, ENCUT, NUPDOWN": "variable parameters",
        "NCORE": 12,
        "KPAR": 2,
        "AEXX": 0.25,
        "ENCUT": 400,
        "ICORELEVEL": "0 # Needed if using the Kumagai-Oba (eFNV) anisotropic charge correction",
        "NSW": 0,
        "LSORBIT": True,
        "EDIFF": 1e-06,  # tight for final energy and converged DOS
        "EDIFFG": -0.01,
        "ALGO": "Normal # Change to All if ZHEGV, FEXCP/F or ZBRENT errors encountered",
        "HFSCREEN": 0.2,
        "IBRION": -1,
        "ISYM": 0,
        "ISMEAR": 0,
        "LASPH": True,
        "LHFCALC": True,
        "LORBIT": 14,
        "LREAL": False,
        "LVHAR": "True # Needed if using the Freysoldt (FNV) charge correction scheme",
        "LWAVE": True,
        "NEDOS": 2000,
        "NELM": 100,
        "PREC": "Accurate",
        "PRECFOCK": "Fast",
        "SIGMA": 0.05,
    }

    # Directory
    vaspnclinputdir = input_dir + "/vasp_ncl/" if input_dir else "VASP_Files/vasp_ncl/"
    if not os.path.exists(vaspnclinputdir):
        os.makedirs(vaspnclinputdir)

    # POTCAR
    potcar_dict = default_potcar_dict
    if potcar_settings:
        if "POTCAR_FUNCTIONAL" in potcar_settings.keys():
            potcar_dict["POTCAR_FUNCTIONAL"] = potcar_settings["POTCAR_FUNCTIONAL"]
        if "POTCAR" in potcar_settings.keys():
            potcar_dict["POTCAR"].update(potcar_settings.pop("POTCAR"))
    vaspnclinput = DictSet(structure, config_dict=potcar_dict)
    vaspnclinput.potcar.write_file(vaspnclinputdir + "POTCAR")

    if all(is_metal(element) for element in structure.composition.elements):
        vaspnclincardict["ISMEAR"] = "2 # Metal, use Methfessel-Paxton smearing scheme"
    if all(is_metal(element) for element in structure.composition.elements):
        vaspnclincardict["SIGMA"] = 0.02

    if incar_settings:
        for (
            k
        ) in (
            incar_settings.keys()
        ):  # check INCAR flags and warn if they don't exist (typos)
            if (
                k not in incar_params.keys()
            ):  # this code is taken from pymatgen.io.vasp.inputs
                warnings.warn(  # but only checking keys, not values so we can add comments etc
                    "Cannot find %s from your user_incar_settings in the list of INCAR flags"
                    % (k),
                    BadIncarWarning,
                )
        vaspnclincardict.update(incar_settings)

    # KPOINTS
    vaspnclkpointsdict = {
        "comment": "Kpoints from doped.vasp_ncl_files",
        "generation_style": "Gamma",  # Set to Monkhorst for Monkhorst-Pack generation
        "kpoints": [[2, 2, 2]],
    }
    if kpoints_settings:
        vaspnclkpointsdict.update(kpoints_settings)
    vaspnclkpts = Kpoints.from_dict(vaspnclkpointsdict)
    vaspnclkpts.write_file(vaspnclinputdir + "KPOINTS")

    # INCAR
    vaspnclincar = Incar.from_dict(vaspnclincardict)
    with zopen(vaspnclinputdir + "INCAR", "wt") as incar_file:
        incar_file.write(vaspnclincar.get_string())
