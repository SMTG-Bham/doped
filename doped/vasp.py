"""
Code to generate VASP defect calculation input files.
"""
import contextlib
import copy
import os
import warnings
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
from monty.json import MSONable
from monty.serialization import dumpfn, loadfn
from pymatgen.core import SETTINGS
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import BadIncarWarning, Kpoints, Poscar, Potcar, incar_params
from pymatgen.io.vasp.sets import DictSet, UserPotcarFunctional
from tqdm import tqdm

from doped import _ignore_pmg_warnings
from doped.core import DefectEntry
from doped.generation import (
    DefectsGenerator,
    _custom_formatwarning,
    _frac_coords_sort_func,
    get_defect_name_from_entry,
    name_defect_entries,
)

# TODO: Go through and update docstrings with descriptions all the default behaviour (INCAR,
#  KPOINTS settings etc)
# TODO: Ensure json serializability, and have optional parameter to output DefectRelaxSet jsons to
#  written folders as well (but off by default)
# TODO: Likewise, add same to/from json, __repr__ etc. functions for DefectRelaxSet. Dict methods apply
#  to `.defect_sets` etc
# TODO: Implement renaming folders like SnB if we try to write a folder that already exists,
#  and the structures don't match (otherwise overwrite)
# TODO: Cleanup; look at refactoring these functions as much as possible to avoid code duplication

_ignore_pmg_warnings()
warnings.formatwarning = _custom_formatwarning


def deep_dict_update(d: dict, u: dict) -> dict:
    """
    Recursively update nested dictionaries without overwriting existing keys.
    """
    for k, v in u.items():
        d[k] = deep_dict_update(d.get(k, {}), v) if isinstance(v, dict) else v
    return d


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
default_potcar_dict = loadfn(os.path.join(MODULE_DIR, "VASP_sets/PotcarSet.yaml"))
default_relax_set = loadfn(os.path.join(MODULE_DIR, "VASP_sets/RelaxSet.yaml"))
default_HSE_set = loadfn(os.path.join(MODULE_DIR, "VASP_sets/HSESet.yaml"))
default_defect_set = loadfn(os.path.join(MODULE_DIR, "VASP_sets/DefectSet.yaml"))
default_defect_relax_set = copy.deepcopy(default_relax_set)
default_defect_relax_set = deep_dict_update(
    default_defect_relax_set, default_defect_set
)  # defect set is just INCAR settings
default_defect_relax_set = deep_dict_update(
    default_defect_relax_set, default_potcar_dict
)  # only POTCAR settings, not set in other *Set.yamls
singleshot_incar_settings = {
    "EDIFF": 1e-6,  # tight EDIFF for final energy and converged DOS
    "NSW": 0,  # no ionic relaxation"
    "IBRION": -1,  # no ionic relaxation"
}


def _test_potcar_functional_choice(potcar_functional):
    """
    Check if the potcar functional choice needs to be changed to match those
    available.
    """
    test_potcar = None
    try:
        test_potcar = Potcar(["Mg"], functional=potcar_functional)
    except OSError as e:
        # try other functional choices:
        if potcar_functional.startswith("PBE"):
            for pbe_potcar_string in ["PBE", "PBE_52", "PBE_54"]:
                with contextlib.suppress(OSError):
                    potcar_functional: UserPotcarFunctional = pbe_potcar_string
                    test_potcar = Potcar(["Mg"], functional=potcar_functional)
                    break

        if test_potcar is None:
            raise e

    return potcar_functional


class DefectDictSet(DictSet):
    """
    Extension to pymatgen DictSet object for VASP defect calculations.
    """

    def __init__(
        self,
        structure: Structure,
        charge_state: int = 0,
        user_incar_settings: Optional[dict] = None,
        user_kpoints_settings: Optional[Union[dict, Kpoints]] = None,
        user_potcar_functional: Optional[UserPotcarFunctional] = "PBE",
        user_potcar_settings: Optional[dict] = None,
        poscar_comment: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            structure (Structure): pymatgen Structure object of the defect supercell
            charge_state (int): Charge of the defect structure
            user_incar_settings (dict):
                Dictionary of user INCAR settings (AEXX, NCORE etc.) to override
                default settings. Highly recommended to look at output INCARs
                or the `RelaxSet.yaml` and `DefectSet.yaml` files in the
                `doped/VASP_sets` folder, to see what the default INCAR settings are.
                Note that any flags that aren't numbers or True/False need to be input
                as strings with quotation marks (e.g. `{"ALGO": "All"}`).
                (default: None)
            user_kpoints_settings (dict or Kpoints):
                Dictionary of user KPOINTS settings (in pymatgen DictSet() format) e.g.,
                {"reciprocal_density": 123}, or a Kpoints object. Default is Gamma-centred,
                reciprocal_density = 100 [Å⁻³].
            user_potcar_functional (str):
                POTCAR functional to use. Default is "PBE" and if this fails,
                tries "PBE_52", then "PBE_54".
            user_potcar_settings (dict):
                Override the default POTCARs, e.g. {"Li": "Li_sv"}. See
                `doped/VASP_sets/PotcarSet.yaml` for the default `POTCAR` set.
            poscar_comment (str):
                Comment line to use for POSCAR files. Default is defect name,
                fractional coordinates of initial site and charge state.
            **kwargs: Additional kwargs to pass to DictSet.
        """
        self.potcars = kwargs.pop("potcars", True)  # to allow testing on GH Actions
        self.charge_state = charge_state
        self.poscar_comment = (
            poscar_comment
            if poscar_comment is not None
            else f"{structure.formula} {'+' if self.charge_state > 0 else ''}{self.charge_state}"
        )

        # get base config and set EDIFF
        relax_set = copy.deepcopy(default_defect_relax_set)
        relax_set["INCAR"]["EDIFF"] = scaled_ediff(len(structure))
        relax_set["INCAR"].pop("EDIFF_PER_ATOM", None)

        lhfcalc = (
            True if user_incar_settings is None else user_incar_settings.get("LHFCALC", True)
        )  # True (hybrid) by default
        if lhfcalc or (isinstance(lhfcalc, str) and lhfcalc.lower().startswith("t")):
            relax_set = deep_dict_update(relax_set, default_HSE_set)  # HSE set is just INCAR settings

        if user_incar_settings is not None:
            for k in user_incar_settings:
                # check INCAR flags and warn if they don't exist (typos)
                if k not in incar_params.keys() and "#" not in k:
                    warnings.warn(  # but only checking keys, not values so we can add comments etc
                        f"Cannot find {k} from your user_incar_settings in the list of INCAR flags",
                        BadIncarWarning,
                    )
            relax_set["INCAR"].update(user_incar_settings)

            if (
                "EDIFFG" not in user_incar_settings
                and relax_set["INCAR"]["IBRION"] == -1
                and relax_set["INCAR"]["NSW"] == 0
            ):
                # singlepoint calc, remove EDIFFG & POTIM from INCAR to avoid confusion:
                for key in ["EDIFFG", "POTIM"]:
                    if key in relax_set["INCAR"]:
                        del relax_set["INCAR"][key]

        # if "length" in user kpoint settings then pop reciprocal_density and use length instead
        if relax_set["KPOINTS"].get("length") or kwargs.get("user_kpoints_settings", {}).get("length"):
            relax_set["KPOINTS"].pop("reciprocal_density", None)

        self.config_dict = self.CONFIG = relax_set  # avoid bug in pymatgen 2023.5.10, PR'd and fixed in
        # later versions

        super(self.__class__, self).__init__(
            structure,
            config_dict=self.config_dict,
            user_incar_settings=user_incar_settings,
            user_kpoints_settings=user_kpoints_settings,
            user_potcar_functional=user_potcar_functional,
            user_potcar_settings=user_potcar_settings,
            force_gamma=kwargs.pop("force_gamma", True),  # force gamma-centred k-points by default
            **kwargs,
        )

    @property
    def incar(self):
        """
        Returns the Incar object generated from the config_dict, with NELECT
        and NUPDOWN set accordingly.
        """
        incar_obj = super(self.__class__, self).incar

        try:
            incar_obj["NELECT"] = self.nelect - self.charge_state
            if incar_obj["NELECT"] % 2 != 0:  # odd number of electrons
                incar_obj["NUPDOWN"] = 1
            else:
                # when writing VASP just resets this to 0 anyway:
                incar_obj["NUPDOWN"] = (
                    "0  # If defect has multiple spin-polarised states (e.g. bipolarons) could "
                    "also have triplet (NUPDOWN=2), but ΔE typically small."
                )

        except Exception as e:
            warnings.warn(
                f"NELECT and NUPDOWN flags are not set due to non-availability of POTCARs; "
                f"got error: {e}"
            )

        if "KPAR" in incar_obj and self.user_kpoints_settings.get("comment", "").startswith("Γ-only"):
            # check KPAR setting is reasonable for number of KPOINTS
            warnings.warn("KPOINTS are Γ-only (i.e. only one kpoint), so KPAR is being set to 1")
            incar_obj["KPAR"] = "1  # Only one k-point (Γ-only)"

        return incar_obj

    @property
    def potcar(self) -> Potcar:
        """
        Potcar object.

        Redefined to intelligently handle pymatgen POTCAR issues.
        """
        if self.potcars:
            self.user_potcar_functional: UserPotcarFunctional = _test_potcar_functional_choice(
                self.user_potcar_functional
            )
        return super(self.__class__, self).potcar

    @property
    def poscar(self) -> Poscar:
        """
        Return Poscar object with comment.
        """
        return Poscar(self.structure, comment=self.poscar_comment)

    @property
    def kpoints(self):
        """
        Return kpoints object with comment.
        """
        pmg_kpoints = super().kpoints
        kpt_density = self.config_dict.get("KPOINTS", {}).get("reciprocal_density", False)
        if (
            isinstance(self.user_kpoints_settings, dict)
            and "reciprocal_density" in self.user_kpoints_settings
        ):
            kpt_density = self.user_kpoints_settings.get("reciprocal_density", False)

        if kpt_density and "doped" not in pmg_kpoints.comment:
            with contextlib.suppress(Exception):
                assert np.prod(pmg_kpoints.kpts[0])  # check if it's a kpoint mesh (not custom kpoints)
                pmg_kpoints.comment = f"KPOINTS from doped, with reciprocal_density = {kpt_density}/Å⁻³"

        elif "doped" not in pmg_kpoints.comment:
            pmg_kpoints.comment = "KPOINTS from doped"

        return pmg_kpoints

    def _check_user_potcars(self, unperturbed_poscar: bool = False) -> bool:
        """
        Check and warn the user if POTCARs are not set up with pymatgen.
        """
        potcars = any("VASP_PSP_DIR" in i for i in SETTINGS)
        if not potcars:
            potcar_warning_string = (
                "POTCAR directory not set up with pymatgen (see the doped docs Installation page: "
                "https://doped.readthedocs.io/en/latest/Installation.html for instructions on setting "
                "this up). This is required to generate `POTCAR` and `INCAR` files (to set `NELECT` and "
                "`NUPDOWN`)"
            )
            if unperturbed_poscar:
                warnings.warn(
                    f"{potcar_warning_string}, so only (unperturbed) `POSCAR` and `KPOINTS` files will "
                    f"be generated."
                )
            else:
                raise ValueError(f"{potcar_warning_string}, so no input files will be generated.")
            return False

        return True

    def write_input(
        self,
        output_dir: str,
        make_dir_if_not_present: bool = True,
        include_cif: bool = False,
        potcar_spec: bool = False,
        zip_output: bool = False,
    ):
        """
        Writes out all input to a directory. Refactored slightly from pymatgen
        DictSet.write_input() to allow checking of user POTCAR setup.

        Copied from DictSet.write_input() docstring:
        Args:
            output_dir (str): Directory to output the VASP input files
            make_dir_if_not_present (bool): Set to True if you want the
                directory (and the whole path) to be created if it is not
                present.
            include_cif (bool): Whether to write a CIF file in the output
                directory for easier opening by VESTA.
            potcar_spec (bool): Instead of writing the POTCAR, write a "POTCAR.spec".
                This is intended to help sharing an input set with people who might
                not have a license to specific Potcar files. Given a "POTCAR.spec",
                the specific POTCAR file can be re-generated using pymatgen with the
                "generate_potcar" function in the pymatgen CLI.
            zip_output (bool): Whether to zip each VASP input file written to the output directory.
        """
        if not potcar_spec:
            self._check_user_potcars(unperturbed_poscar=True)

        super().write_input(
            output_dir,
            make_dir_if_not_present=make_dir_if_not_present,
            include_cif=include_cif,
            potcar_spec=potcar_spec,
            zip_output=zip_output,
        )


def scaled_ediff(natoms: int) -> float:
    """
    Returns a scaled EDIFF value for VASP calculations, based on the number of
    atoms in the structure.

    EDIFF is set to 2e-7 per atom (-> 1e-5 per 50 atoms), with a maximum EDIFF
    of 1e-4.
    """
    ediff = float(f"{((natoms/50)*1e-5):.1g}")
    return min(ediff, 1e-4)


class DefectRelaxSet(MSONable):
    """
    An object for generating input files for VASP defect relaxation
    calculations from pymatgen `DefectEntry` (recommended) or `Structure`
    objects.

    The supercell structure and charge state are taken from the `DefectEntry`
    attributes, or if a `Structure` is provided, then from the
    `defect_supercell` and `charge_state` input parameters.
    """

    def __init__(
        self,
        defect_entry: Union[DefectEntry, Structure],
        charge_state: Optional[int] = None,
        soc: Optional[bool] = None,
        user_incar_settings: Optional[dict] = None,
        user_kpoints_settings: Optional[Union[dict, Kpoints]] = None,
        user_potcar_functional: Optional[UserPotcarFunctional] = "PBE",
        user_potcar_settings: Optional[dict] = None,
        **kwargs,
    ):
        """
        Creates attributes:
        - `DefectRelaxSet.vasp_gam` -> `DefectDictSet` for Gamma-point only
            relaxation. Not needed if ShakeNBreak structure searching has been
            performed (recommended), unless only Γ-point _k_-point sampling is
            required (converged) for your system, and no vasp_std calculations
            with multiple _k_-points are required (determined from kpoints settings).
        - `DefectRelaxSet.vasp_nkred_std` -> `DefectDictSet` for relaxation with a
            kpoint mesh and using `NKRED`. Not generated for GGA calculations (if
            `LHFCALC` is set to `False` in user_incar_settings) or if only Gamma
            kpoint sampling is required.
        - `DefectRelaxSet.vasp_std` -> `DefectDictSet` for relaxation with a kpoint
            mesh, not using `NKRED`. Not generated if only Gamma kpoint sampling is
            required.
        - `DefectRelaxSet.vasp_ncl` -> `DefectDictSet` for singlepoint (static)
            energy calculation with SOC included. Generated if `soc=True`. If `soc`
            is not set, then by default is only generated for systems with a max
            atomic number (Z) >= 31 (i.e. further down the periodic table than Zn).
        where `DefectDictSet` is an extension of `pymatgen`'s `DictSet` class for
        defect calculations, with `incar`, `poscar`, `kpoints` and `potcar`
        attributes for the corresponding VASP defect calculations (see docstring).
        Also creates the corresponding `bulk_vasp_..` attributes for singlepoint
        (static) energy calculations of the bulk (pristine, defect-free)
        supercell. This needs to be calculated once with the same settings as the
        defect calculations, for the later calculation of defect formation energies.

        See the `RelaxSet.yaml` and `DefectSet.yaml` files in the
        `doped/VASP_sets` folder for the default `INCAR` and `KPOINT` settings,
        and `PotcarSet.yaml` for the default `POTCAR` settings.
        **These are reasonable defaults that _roughly_ match the typical values
        needed for accurate defect calculations, but usually will need to be
        modified for your specific system, such as converged ENCUT and KPOINTS,
        and NCORE / KPAR matching your HPC setup.**

        Note that any changes to the default `INCAR`/`POTCAR` settings should
        be consistent with those used for all defect and competing phase
        (chemical potential) calculations.

        Args:
            defect_entry (DefectEntry, Structure):
                doped/pymatgen DefectEntry or Structure (defect supercell) for
                which to generate `DefectDictSet`s for.
            charge_state (int):
                Charge state of the defect. Overrides `DefectEntry.charge_state` if
                `DefectEntry` is input.
            soc (bool):
                Whether to generate `vasp_ncl` DefectDictSet attribute for spin-orbit
                coupling singlepoint (static) energy calculations. If not set, then
                by default is set to True if the max atomic number (Z) in the
                structure is >= 31 (i.e. further down the periodic table than Zn),
                otherwise False.
            user_incar_settings (dict):
                Dictionary of user INCAR settings (AEXX, NCORE etc.) to override
                default settings. Highly recommended to look at output INCARs
                or the `RelaxSet.yaml` and `DefectSet.yaml` files in the
                `doped/VASP_sets` folder, to see what the default INCAR settings are.
                Note that any flags that aren't numbers or True/False need to be input
                as strings with quotation marks (e.g. `{"ALGO": "All"}`).
                (default: None)
            user_kpoints_settings (dict or Kpoints):
                Dictionary of user KPOINTS settings (in pymatgen DictSet() format)
                e.g. {"reciprocal_density": 123}, or a Kpoints object, to use for the
                `vasp_std`, `vasp_nkred_std` and `vasp_ncl` DefectDictSets (Γ-only for
                `vasp_gam`). Default is Gamma-centred, reciprocal_density = 100 [Å⁻³].
            user_potcar_functional (str):
                POTCAR functional to use. Default is "PBE" and if this fails,
                tries "PBE_52", then "PBE_54".
            user_potcar_settings (dict):
                Override the default POTCARs, e.g. {"Li": "Li_sv"}. See
                `doped/VASP_sets/PotcarSet.yaml` for the default `POTCAR` set.
            **kwargs: Additional kwargs to pass to DictSet().

        Attributes:
            vasp_gam (DefectDictSet):
                DefectDictSet for Gamma-point only relaxation. Not needed if
                ShakeNBreak structure searching has been performed (recommended),
                unless only Γ-point _k_-point sampling is required (converged)
                for your system, and no vasp_std calculations with multiple
                _k_-points are required (determined from kpoints settings).
            vasp_nkred_std (DefectDictSet):
                DefectDictSet for relaxation with a non-Γ-only kpoint mesh, using
                `NKRED(X,Y,Z)` INCAR tag(s) to downsample kpoints for the HF exchange
                part of the hybrid DFT calculation`. Not generated for GGA calculations
                (if `LHFCALC` is set to `False` in user_incar_settings) or if only Gamma
                kpoint sampling is required.
            vasp_std (DefectDictSet):
                DefectDictSet for relaxation with a non-Γ-only kpoint mesh, not using
                `NKRED`. Not generated if only Gamma kpoint sampling is required.
            vasp_ncl (DefectDictSet):
                DefectDictSet for singlepoint (static) energy calculation with SOC
                included. Generated if `soc=True`. If `soc` is not set, then by default
                is only generated for systems with a max atomic number (Z) >= 31
                (i.e. further down the periodic table than Zn).
            defect_supercell (Structure):
                Supercell structure for defect calculations, taken from
                defect_entry.defect_supercell (if defined), otherwise from
                defect_entry.sc_entry.structure if inputting a DefectEntry object,
                or the input structure if inputting a Structure object.
            bulk_supercell (Structure):
                Supercell structure of the bulk (pristine, defect-free) material,
                taken from defect_entry.bulk_supercell (if defined), otherwise from
                defect_entry.bulk_entry.structure if inputting a DefectEntry object,
                or None if inputting a Structure object.
            poscar_comment (str):
                Comment to write at the top of the POSCAR files. Default is the
                defect entry name, defect frac coords and charge state (if
                inputting a DefectEntry object), or the formula of the input
                structure and charge state (if inputting a Structure object),
                for defects. For the bulk supercell, it's "{formula} - Bulk".
            bulk_vasp_gam (DefectDictSet):
                DefectDictSet for a _bulk_ Γ-point-only singlepoint (static)
                supercell calculation. Often not used, as the bulk supercell only
                needs to be calculated once with the same settings as the final
                defect calculations, which may be with `vasp_std` or `vasp_ncl`.
            bulk_vasp_nkred_std (DefectDictSet):
                DefectDictSet for a singlepoint (static) _bulk_ `vasp_std` supercell
                calculation (i.e. with a non-Γ-only kpoint mesh) and `NKRED(X,Y,Z)`
                INCAR tag(s) to downsample kpoints for the HF exchange part of the
                hybrid DFT calculation. Not generated for GGA calculations (if
                `LHFCALC` is set to `False` in user_incar_settings) or if only Gamma
                kpoint sampling is required.
            bulk_vasp_std (DefectDictSet):
                DefectDictSet for a singlepoint (static) _bulk_ `vasp_std` supercell
                calculation with a non-Γ-only kpoint mesh, not using `NKRED`. Not
                generated if only Gamma kpoint sampling is required.
            bulk_vasp_ncl (DefectDictSet):
                DefectDictSet for singlepoint (static) energy calculation of the _bulk_
                supercell with SOC included. Generated if `soc=True`. If `soc` is not
                set, then by default is only generated for systems with a max atomic
                number (Z) >= 31 (i.e. further down the periodic table than Zn).

            Input parameters are also set as attributes.
        """
        self.defect_entry = defect_entry
        self.charge_state = charge_state or self.defect_entry.charge_state
        self.user_incar_settings = user_incar_settings or {}
        self.user_kpoints_settings = user_kpoints_settings or {}
        self.user_potcar_functional = user_potcar_functional
        self.user_potcar_settings = user_potcar_settings or {}
        self.dict_set_kwargs = kwargs or {}

        if isinstance(self.defect_entry, Structure):
            self.defect_supercell = self.defect_entry
            self.poscar_comment = (
                f"{self.defect_supercell.formula} {'+' if self.charge_state > 0 else ''}"
                f"{self.charge_state}"
            )
            self.bulk_supercell = None

        elif isinstance(self.defect_entry, DefectEntry):
            self.defect_supercell = (
                self.defect_entry.defect_supercell or self.defect_entry.sc_entry.structure
            )
            if self.defect_entry.bulk_supercell is not None:
                self.bulk_supercell = self.defect_entry.bulk_supercell
            elif self.defect_entry.bulk_entry is not None:
                self.bulk_supercell = self.defect_entry.bulk_entry.structure
            else:
                raise ValueError(
                    "Bulk supercell must be defined in DefectEntry object attributes. Both "
                    "DefectEntry.bulk_supercell and DefectEntry.bulk_entry are None!"
                )

            # get POSCAR comment:
            sc_frac_coords = self.defect_entry.sc_defect_frac_coords
            if sc_frac_coords is None:
                raise ValueError(
                    "Fractional coordinates of defect in the supercell must be defined in "
                    "DefectEntry object attributes, but DefectEntry.sc_defect_frac_coords "
                    "is None!"
                )
            approx_coords = f"~[{sc_frac_coords[0]:.4f},{sc_frac_coords[1]:.4f},{sc_frac_coords[2]:.4f}]"
            # Gets truncated to 40 characters in the CONTCAR (so kept to less than 40 chars here)
            if hasattr(self.defect_entry, "name"):
                name = self.defect_entry.name.rsplit("_", 1)[0]  # remove charge state from name
            else:
                name = self.defect_supercell.formula

            self.poscar_comment = (
                f"{name} {approx_coords} {'+' if self.charge_state > 0 else ''}{self.charge_state}"
            )

        else:
            raise TypeError("defect_entry must be a doped/pymatgen DefectEntry or Structure object.")

        if soc is not None:
            self.soc = soc
        else:
            self.soc = max(self.defect_supercell.atomic_numbers) >= 31  # use defect supercell rather
            # than defect.defect_structure because could be e.g. a vacancy in a 2-atom primitive
            # structure where the atom being removed is the heavy (Z>=31) one

    @property
    def vasp_gam(
        self,
    ) -> DefectDictSet:
        """
        Returns a DefectDictSet object for a VASP Γ-point-only (`vasp_gam`)
        defect supercell relaxation. Typically not needed if ShakeNBreak
        structure searching has been performed (recommended), unless only
        Γ-point _k_-point sampling is required (converged) for your system, and
        no vasp_std calculations with multiple _k_-points are required
        (determined from kpoints settings).

        See the `RelaxSet.yaml` and `DefectSet.yaml` files in the
        `doped/VASP_sets` folder for the default `INCAR` and `KPOINT` settings,
        and `PotcarSet.yaml` for the default `POTCAR` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default `INCAR`/`POTCAR` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.
        """
        return DefectDictSet(
            self.defect_supercell,
            charge_state=self.defect_entry.charge_state,
            user_incar_settings=self.user_incar_settings,
            user_kpoints_settings=Kpoints().from_dict(
                {
                    "comment": "Γ-only KPOINTS from doped",
                    "generation_style": "Gamma",
                }
            ),
            user_potcar_functional=self.user_potcar_functional,
            user_potcar_settings=self.user_potcar_settings,
            poscar_comment=self.poscar_comment,
            **self.dict_set_kwargs,
        )

    def _check_vstd_kpoints(self, vasp_std_defect_set: DefectDictSet) -> Optional[DefectDictSet]:
        try:
            vasp_std = None if np.prod(vasp_std_defect_set.kpoints.kpts[0]) == 1 else vasp_std_defect_set
        except Exception:  # different kpoint generation scheme, not Gamma-only
            vasp_std = vasp_std_defect_set

        if vasp_std is None:
            current_kpoint_settings = (
                ("default `reciprocal_density = 100` [Å⁻³] (see doped/VASP_sets/RelaxSet.yaml)")
                if self.user_kpoints_settings is None
                else self.user_kpoints_settings
            )
            warnings.warn(
                f"With the current kpoint settings ({current_kpoint_settings}), the k-point mesh is "
                f"Γ-only (see docstrings). Thus vasp_std should not be used, and vasp_gam should be used "
                f"instead."
            )

        return vasp_std

    @property
    def vasp_std(self) -> Optional[DefectDictSet]:
        """
        Returns a DefectDictSet object for a VASP defect supercell relaxation
        using `vasp_std` (i.e. with a non-Γ-only kpoint mesh). Returns None and
        a warning if the input kpoint settings correspond to a Γ-only kpoint
        mesh (in which case `vasp_gam` should be used).

        See the `RelaxSet.yaml` and `DefectSet.yaml` files in the
        `doped/VASP_sets` folder for the default `INCAR` and `KPOINT` settings,
        and `PotcarSet.yaml` for the default `POTCAR` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default `INCAR`/`POTCAR` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.
        """
        user_incar_settings = copy.deepcopy(self.user_incar_settings)
        if "KPAR" not in user_incar_settings:
            user_incar_settings["KPAR"] = 2  # multiple k-points, likely quicker with this

        # determine if vasp_std required or only vasp_gam:
        std_defect_set = DefectDictSet(
            self.defect_supercell,
            charge_state=self.defect_entry.charge_state,
            user_incar_settings=user_incar_settings,
            user_kpoints_settings=self.user_kpoints_settings,
            user_potcar_functional=self.user_potcar_functional,
            user_potcar_settings=self.user_potcar_settings,
            poscar_comment=self.poscar_comment,
            **self.dict_set_kwargs,
        )

        return self._check_vstd_kpoints(std_defect_set)

    @property
    def vasp_nkred_std(self) -> Optional[DefectDictSet]:
        """
        Returns a DefectDictSet object for a VASP defect supercell relaxation
        using `vasp_std` (i.e. with a non-Γ-only kpoint mesh) and
        `NKRED(X,Y,Z)` INCAR tag(s) to downsample kpoints for the HF exchange
        part of hybrid DFT calculations, following the doped recommended defect
        calculation workflow (see docs). By default, sets `NKRED(X,Y,Z)` to 2
        or 3 in the directions for which the k-point grid is divisible by this
        factor. Returns None and a warning if the input kpoint settings
        correspond to a Γ-only kpoint mesh (in which case `vasp_gam` should be
        used) or for GGA calculations (if `LHFCALC` is set to `False` in
        user_incar_settings, in which case `vasp_std` should be used).

        See the `RelaxSet.yaml` and `DefectSet.yaml` files in the
        `doped/VASP_sets` folder for the default `INCAR` and `KPOINT` settings,
        and `PotcarSet.yaml` for the default `POTCAR` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default `INCAR`/`POTCAR` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.
        """
        if self.user_incar_settings.get("LHFCALC", True) is False:  # GGA
            warnings.warn(
                "`LHFCALC` is set to `False` in user_incar_settings, so `vasp_nkred_std` (as NKRED "
                "acts to downsample the Fock exchange potential in _hybrid DFT_ calculations), "
                "and so `vasp_std` should be used instead."
            )
            return None

        std_defect_set = self.vasp_std

        if std_defect_set is not None:  # non Γ-only kpoint mesh
            # determine appropriate NKRED:
            try:
                kpt_mesh = np.array(std_defect_set.kpoints.kpts[0])
                # if all divisible by 2 or 3, then set NKRED to 2 or 3, respectively:
                nkred_dict = {
                    "NKRED": None,
                    "NKREDX": None,
                    "NKREDY": None,
                    "NKREDZ": None,
                }  # type: Dict[str, Optional[int]]
                for k in [2, 3]:
                    if np.all(kpt_mesh % k == 0):
                        nkred_dict["NKRED"] = k
                        break

                    for idx, nkred_key in enumerate(["NKREDX", "NKREDY", "NKREDZ"]):
                        if kpt_mesh[idx] % k == 0:
                            nkred_dict[nkred_key] = k
                            break

                incar_settings = copy.deepcopy(std_defect_set.user_incar_settings)  # so KPAR also included
                if nkred_dict["NKRED"] is not None:
                    incar_settings["NKRED"] = nkred_dict["NKRED"]
                else:
                    for nkred_key in ["NKREDX", "NKREDY", "NKREDZ"]:
                        if nkred_dict[nkred_key] is not None:
                            incar_settings[nkred_key] = nkred_dict[nkred_key]

            except Exception:
                warnings.warn(
                    f"The specified kpoint settings ({self.user_kpoints_settings,}) do not give a "
                    f"grid-like k-point mesh and so the appropriate NKRED settings cannot be "
                    f"automatically determined. Either set NKRED manually using `user_incar_settings` "
                    f"with `vasp_std`/`write_std`, or adjust your k-point settings."
                )
                return None

            return DefectDictSet(
                self.defect_supercell,
                charge_state=self.defect_entry.charge_state,
                user_incar_settings=incar_settings,
                user_kpoints_settings=self.user_kpoints_settings,
                user_potcar_functional=self.user_potcar_functional,
                user_potcar_settings=self.user_potcar_settings,
                poscar_comment=self.poscar_comment,
                **self.dict_set_kwargs,
            )
        return None

    @property
    def vasp_ncl(self) -> Optional[DefectDictSet]:
        """
        Returns a DefectDictSet object for a VASP defect supercell singlepoint
        calculation with spin-orbit coupling (SOC) included (LSORBIT = True),
        using `vasp_ncl`. If `DefectRelaxSet.soc` is False, then this returns
        None and a warning. If the `soc` parameter is not set when initializing
        `DefectRelaxSet`, then this is set to True for systems with a max
        atomic number (Z) >= 31 (i.e. further down the periodic table than Zn),
        otherwise False.

        See the `RelaxSet.yaml` and `DefectSet.yaml` files in the
        `doped/VASP_sets` folder for the default `INCAR` and `KPOINT` settings,
        and `PotcarSet.yaml` for the default `POTCAR` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default `INCAR`/`POTCAR` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.
        """
        if not self.soc:
            warnings.warn(
                "DefectRelaxSet.soc is False, so vasp_ncl is None (i.e. no `vasp_ncl` input files have "
                "been generated). If SOC calculations are desired, set soc=True when initializing "
                "DefectRelaxSet. Otherwise, use vasp_std or vasp_gam instead."
            )
            return None

        user_incar_settings = copy.deepcopy(self.user_incar_settings)
        user_incar_settings.update(singleshot_incar_settings)
        user_incar_settings.update({"LSORBIT": True})  # ISYM already 0

        if "KPAR" not in user_incar_settings:
            user_incar_settings["KPAR"] = 2  # likely quicker with this, checked in DefectDictSet
            # if it's Γ-only vasp_ncl and reset to 1 accordingly

        # ignore warning with "KPAR", in case it's Γ-only
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "KPAR")
            return DefectDictSet(
                self.defect_supercell,
                charge_state=self.defect_entry.charge_state,
                user_incar_settings=user_incar_settings,
                user_kpoints_settings=self.user_kpoints_settings,
                user_potcar_functional=self.user_potcar_functional,
                user_potcar_settings=self.user_potcar_settings,
                poscar_comment=self.poscar_comment,
                **self.dict_set_kwargs,
            )

    def _check_bulk_supercell_and_warn(self):
        if self.bulk_supercell is None:
            warnings.warn(
                "DefectRelaxSet.bulk_supercell is None because a Structure object rather than "
                "DefectEntry was provided (see docstring), and so bulk supercell files cannot "
                "be generated this way. Either provide a DefectEntry or manually copy and edit "
                "the input files from defect calculations to use for the bulk supercell "
                "reference calculation."
            )
            return None

        return self.bulk_supercell

    @property
    def bulk_vasp_gam(self) -> Optional[DefectDictSet]:
        """
        Returns a DefectDictSet object for a VASP _bulk_ Γ-point-only
        (`vasp_gam`) singlepoint (static) supercell calculation. Often not
        used, as the bulk supercell only needs to be calculated once with the
        same settings as the final defect calculations, which is `vasp_std` if
        we have a non-Γ-only final k-point mesh, or `vasp_ncl` if SOC effects
        are being included. If the final converged k-point mesh is Γ-only, then
        `bulk_vasp_gam` should be used to calculate the singlepoint (static)
        bulk supercell reference energy. Can also sometimes be useful for the
        purpose of calculating defect formation energies at early stages of the
        typical `vasp_gam` -> `vasp_nkred_std` (if hybrid & non-Γ-only
        k-points) -> `vasp_std` (if non-Γ-only k-points) -> `vasp_ncl` (if SOC
        included) workflow, to obtain rough formation energy estimates and flag
        any potential issues with defect calculations early on.

        See the `RelaxSet.yaml` and `DefectSet.yaml` files in the
        `doped/VASP_sets` folder for the default `INCAR` and `KPOINT` settings,
        and `PotcarSet.yaml` for the default `POTCAR` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default `INCAR`/`POTCAR` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.
        """
        bulk_supercell: Structure = self._check_bulk_supercell_and_warn()
        if bulk_supercell is None:
            return None

        user_incar_settings = copy.deepcopy(self.user_incar_settings)
        user_incar_settings.update(singleshot_incar_settings)

        return DefectDictSet(
            bulk_supercell,
            charge_state=0,
            user_incar_settings=user_incar_settings,
            user_kpoints_settings=Kpoints().from_dict(
                {
                    "comment": "Γ-only KPOINTS from doped",
                    "generation_style": "Gamma",
                }
            ),
            user_potcar_functional=self.user_potcar_functional,
            user_potcar_settings=self.user_potcar_settings,
            poscar_comment=f"{bulk_supercell.formula} - Bulk",
            **self.dict_set_kwargs,
        )

    @property
    def bulk_vasp_std(self) -> Optional[DefectDictSet]:
        """
        Returns a DefectDictSet object for a singlepoint (static) _bulk_
        `vasp_std` supercell calculation. Returns None and a warning if the
        input kpoint settings correspond to a Γ-only kpoint mesh (in which case
        `(bulk_)vasp_gam` should be used).

        The bulk supercell only needs to be calculated once with the same
        settings as the final defect calculations, which is `vasp_std` if we
        have a non-Γ-only final k-point mesh, `vasp_ncl` if SOC effects are
        being included (in which case `bulk_vasp_ncl` should be used for the
        singlepoint bulk supercell reference calculation), or `vasp_gam` if the
        final converged k-point mesh is Γ-only (in which case `bulk_vasp_gam`
        should be used for the singlepoint bulk supercell reference
        calculation). Can also sometimes be useful for the purpose of
        calculating defect formation energies at midway stages of the typical
        `vasp_gam` -> `vasp_nkred_std` (if hybrid & non-Γ-only k-points) ->
        `vasp_std` (if non-Γ-only k-points) -> `vasp_ncl` (if SOC included)
        workflow, to obtain rough formation energy estimates and flag any
        potential issues with defect calculations early on.

        See the `RelaxSet.yaml` and `DefectSet.yaml` files in the
        `doped/VASP_sets` folder for the default `INCAR` and `KPOINT` settings,
        and `PotcarSet.yaml` for the default `POTCAR` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default `INCAR`/`POTCAR` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.
        """
        bulk_supercell: Structure = self._check_bulk_supercell_and_warn()
        if bulk_supercell is None:
            return None

        user_incar_settings = copy.deepcopy(self.user_incar_settings)
        user_incar_settings.update(singleshot_incar_settings)

        if "KPAR" not in user_incar_settings:
            user_incar_settings["KPAR"] = 2  # multiple k-points, likely quicker with this

        return self._check_vstd_kpoints(
            DefectDictSet(
                bulk_supercell,
                charge_state=0,
                user_incar_settings=user_incar_settings,
                user_kpoints_settings=self.user_kpoints_settings,
                user_potcar_functional=self.user_potcar_functional,
                user_potcar_settings=self.user_potcar_settings,
                poscar_comment=f"{bulk_supercell.formula} - Bulk",
                **self.dict_set_kwargs,
            )
        )

    @property
    def bulk_vasp_nkred_std(self) -> Optional[DefectDictSet]:
        """
        Returns a DefectDictSet object for a singlepoint (static) _bulk_
        `vasp_std` supercell calculation (i.e. with a non-Γ-only kpoint mesh)
        and `NKRED(X,Y,Z)` INCAR tag(s) to downsample kpoints for the HF
        exchange part of the hybrid DFT calculation. By default, sets
        `NKRED(X,Y,Z)` to 2 or 3 in the directions for which the k-point grid
        is divisible by this factor. Returns None and a warning if the input
        kpoint settings correspond to a Γ-only kpoint mesh (in which case
        `(bulk_)vasp_gam` should be used) or for GGA calculations (if `LHFCALC`
        is set to `False` in user_incar_settings, in which case
        `(bulk_)vasp_std` should be used).

        The bulk supercell only needs to be calculated once with the same
        settings as the final defect calculations, which is `vasp_std` if we
        have a non-Γ-only final k-point mesh, `vasp_ncl` if SOC effects are
        being included (in which case `bulk_vasp_ncl` should be used for the
        singlepoint bulk supercell reference calculation), or `vasp_gam` if the
        final converged k-point mesh is Γ-only (in which case `bulk_vasp_gam`
        should be used for the singlepoint bulk supercell reference
        calculation). Can also sometimes be useful for the purpose of
        calculating defect formation energies at midway stages of the typical
        `vasp_gam` -> `vasp_nkred_std` (if hybrid & non-Γ-only k-points) ->
        `vasp_std` (if non-Γ-only k-points) -> `vasp_ncl` (if SOC included)
        workflow, to obtain rough formation energy estimates and flag any
        potential issues with defect calculations early on.

        See the `RelaxSet.yaml` and `DefectSet.yaml` files in the
        `doped/VASP_sets` folder for the default `INCAR` and `KPOINT` settings,
        and `PotcarSet.yaml` for the default `POTCAR` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default `INCAR`/`POTCAR` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.
        """
        bulk_supercell: Structure = self._check_bulk_supercell_and_warn()
        if bulk_supercell is None:
            return None

        # check NKRED by running through `vasp_nkred_std`:
        nkred_defect_dict_set = self.vasp_nkred_std

        if nkred_defect_dict_set is None:
            return None

        user_incar_settings = copy.deepcopy(self.user_incar_settings)
        if "KPAR" not in user_incar_settings:
            user_incar_settings["KPAR"] = 2  # multiple k-points, likely quicker with this
        user_incar_settings.update(singleshot_incar_settings)
        user_incar_settings.update(  # add NKRED settings
            {k: v for k, v in nkred_defect_dict_set.incar.as_dict().items() if "NKRED" in k}
        )

        return DefectDictSet(
            bulk_supercell,
            charge_state=0,
            user_incar_settings=user_incar_settings,
            user_kpoints_settings=self.user_kpoints_settings,
            user_potcar_functional=self.user_potcar_functional,
            user_potcar_settings=self.user_potcar_settings,
            poscar_comment=f"{bulk_supercell.formula} - Bulk",
            **self.dict_set_kwargs,
        )

    @property
    def bulk_vasp_ncl(self) -> Optional[DefectDictSet]:
        """
        Returns a DefectDictSet object for VASP _bulk_ supercell singlepoint
        calculations with spin-orbit coupling (SOC) included (LSORBIT = True),
        using `vasp_ncl`. If `DefectRelaxSet.soc` is False, then this returns
        None and a warning. If the `soc` parameter is not set when initializing
        `DefectRelaxSet`, then this is set to True for systems with a max
        atomic number (Z) >= 31 (i.e. further down the periodic table than Zn),
        otherwise False.

        See the `RelaxSet.yaml` and `DefectSet.yaml` files in the
        `doped/VASP_sets` folder for the default `INCAR` and `KPOINT` settings,
        and `PotcarSet.yaml` for the default `POTCAR` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default `INCAR`/`POTCAR` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.
        """
        bulk_supercell: Structure = self._check_bulk_supercell_and_warn()
        if bulk_supercell is None:
            return None

        if not self.soc:
            warnings.warn(
                "DefectRelaxSet.soc is False, so bulk_vasp_ncl is None (i.e. no `vasp_ncl` input files "
                "have been generated). If SOC calculations are desired, set soc=True when initializing "
                "DefectRelaxSet. Otherwise, use bulk_vasp_std or bulk_vasp_gam instead."
            )
            return None

        user_incar_settings = copy.deepcopy(self.user_incar_settings)
        user_incar_settings.update(singleshot_incar_settings)
        user_incar_settings.update({"LSORBIT": True})  # ISYM already 0

        if "KPAR" not in user_incar_settings:
            user_incar_settings["KPAR"] = 2  # likely quicker with this, checked in DefectDictSet
            # if it's Γ-only vasp_ncl and reset to 1 accordingly

        # ignore warning with "KPAR", in case it's Γ-only
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "KPAR")
            return DefectDictSet(
                bulk_supercell,
                charge_state=0,
                user_incar_settings=user_incar_settings,
                user_kpoints_settings=self.user_kpoints_settings,
                user_potcar_functional=self.user_potcar_functional,
                user_potcar_settings=self.user_potcar_settings,
                poscar_comment=f"{bulk_supercell.formula} - Bulk",
                **self.dict_set_kwargs,
            )

    def _get_output_dir(self, defect_dir: Optional[str] = None, subfolder: Optional[str] = None):
        if defect_dir is None:
            if self.defect_entry.name is None:
                self.defect_entry.name = get_defect_name_from_entry(self.defect_entry)

            defect_dir = self.defect_entry.name

        return f"{defect_dir}/{subfolder}" if subfolder is not None else defect_dir

    def _check_potcars_and_write_vasp_xxx_files(
        self, defect_dir, subfolder, unperturbed_poscar, vasp_xxx_attribute, **kwargs
    ):
        output_dir = self._get_output_dir(defect_dir, subfolder)

        if unperturbed_poscar:
            vasp_xxx_attribute.write_input(
                output_dir,
                **kwargs,  # kwargs to allow POTCAR testing on GH Actions
            )

        else:  # use `write_file()`s rather than `write_input()` to avoid writing POSCARs
            os.makedirs(output_dir, exist_ok=True)
            if not kwargs.get("potcar_spec", False):
                vasp_xxx_attribute._check_user_potcars(unperturbed_poscar=False)
            vasp_xxx_attribute.incar.write_file(f"{output_dir}/INCAR")
            vasp_xxx_attribute.kpoints.write_file(f"{output_dir}/KPOINTS")
            if self.user_potcar_functional is not None:  # for GH Actions testing
                vasp_xxx_attribute.potcar.write_file(f"{output_dir}/POTCAR")

        if "bulk" not in defect_dir:  # not a bulk supercell
            self.defect_entry.to_json(f"{output_dir}/{self.defect_entry.name}.json")

    def write_gam(
        self,
        defect_dir: Optional[str] = None,
        subfolder: Optional[str] = "vasp_gam",
        bulk: bool = False,
        **kwargs,
    ):
        """
        Write the input files for VASP Γ-point-only (`vasp_gam`) defect
        supercell relaxation. Typically not recommended for use, as the
        recommended workflow is to perform `vasp_gam` calculations using
        `ShakeNBreak` for defect structure-searching and initial relaxations,
        but should be used if the final, converged _k_-point mesh is Γ-point-
        only. If bulk is True, the input files for a singlepoint calculation of
        the bulk supercell are also written to "{formula}_bulk/{subfolder}".

        See the `RelaxSet.yaml` and `DefectSet.yaml` files in the
        `doped/VASP_sets` folder for the default `INCAR` and `KPOINT` settings,
        and `PotcarSet.yaml` for the default `POTCAR` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default `INCAR`/`POTCAR` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.

        The `DefectEntry` object is also written to a `json` file in
        `defect_dir` to aid calculation provenance.

        Args:
            defect_dir (str):
                Folder in which to create the VASP defect calculation inputs.
                Default is to use the DefectEntry name (e.g. "Y_i_C4v_O1.92_+2"
                etc.), from `self.defect_entry.name`. If this attribute is not
                set, it is automatically generated according to the doped
                convention (using `get_defect_name_from_entry()`).
            subfolder (str):
                Output folder structure is `<defect_dir>/<subfolder>` where
                `subfolder` = 'vasp_gam' by default. Setting `subfolder` to
                `None` will write the `vasp_gam` input files directly to the
                `<defect_dir>` folder, with no subfolders created.
            bulk (bool):
                If True, the input files for a singlepoint calculation of the
                bulk supercell are also written to "{formula}_bulk/{subfolder}".
                (Default: False)
            **kwargs:
                Keyword arguments to pass to `DefectDictSet.write_input()`.
        """
        if defect_dir is None:
            defect_dir = self.defect_entry.name

        self._check_potcars_and_write_vasp_xxx_files(
            defect_dir,
            subfolder,
            unperturbed_poscar=True,
            vasp_xxx_attribute=self.vasp_gam,
            **kwargs,
        )
        if bulk:
            bulk_supercell = self._check_bulk_supercell_and_warn()
            if bulk_supercell is None:
                return

            formula = bulk_supercell.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
            output_dir = os.path.dirname(defect_dir) if "/" in defect_dir else "."
            self._check_potcars_and_write_vasp_xxx_files(
                f"{output_dir}/{formula}_bulk",
                subfolder,
                unperturbed_poscar=True,
                vasp_xxx_attribute=self.bulk_vasp_gam,
                **kwargs,
            )

    def write_std(
        self,
        defect_dir: Optional[str] = None,
        subfolder: Optional[str] = "vasp_std",
        unperturbed_poscar: bool = False,
        bulk: bool = False,
        **kwargs,
    ):
        """
        Write the input files for a VASP defect supercell calculation using
        `vasp_std` (i.e. with a non-Γ-only kpoint mesh). By default, does not
        generate `POSCAR` (input structure) files, as these should be taken
        from the `CONTCAR`s of `vasp_std` relaxations using `NKRED(X,Y,Z)`
        (originally from `ShakeNBreak` relaxations) if using hybrid DFT, or
        from `ShakeNBreak` calculations (via `snb-groundstate`) if using GGA,
        or, if not following the recommended structure-searching workflow, from
        the `CONTCAR`s of `vasp_gam` calculations. If unperturbed `POSCAR`
        files are desired, set `unperturbed_poscar=True`. If bulk is True, the
        input files for a singlepoint calculation of the bulk supercell are
        also written to "{formula}_bulk/{subfolder}".

        Returns None and a warning if the input kpoint settings correspond to
        a Γ-only kpoint mesh (in which case `vasp_gam` should be used).

        See the `RelaxSet.yaml` and `DefectSet.yaml` files in the
        `doped/VASP_sets` folder for the default `INCAR` and `KPOINT` settings,
        and `PotcarSet.yaml` for the default `POTCAR` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default `INCAR`/`POTCAR` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.

        The `DefectEntry` object is also written to a `json` file in
        `defect_dir` to aid calculation provenance.

        Args:
            defect_dir (str):
                Folder in which to create the VASP defect calculation inputs.
                Default is to use the DefectEntry name (e.g. "Y_i_C4v_O1.92_+2"
                etc.), from `self.defect_entry.name`. If this attribute is not
                set, it is automatically generated according to the doped
                convention (using `get_defect_name_from_entry()`).
            subfolder (str):
                Output folder structure is `<defect_dir>/<subfolder>` where
                `subfolder` = 'vasp_std' by default. Setting `subfolder` to
                `None` will write the `vasp_std` input files directly to the
                `<defect_dir>` folder, with no subfolders created.
            unperturbed_poscar (bool):
                If True, write the unperturbed defect POSCARs to the generated
                folders as well. Not recommended, as the recommended workflow is
                to initially perform `vasp_gam` ground-state structure searching
                using ShakeNBreak (https://shakenbreak.readthedocs.io), then
                continue the `vasp_std` relaxations from the 'Groundstate'
                `CONTCAR`s (first with NKRED if using hybrid DFT, with the
                `write_nkred_std()` method, then without NKRED).
                (default: False)
            bulk (bool):
                If True, the input files for a singlepoint calculation of the
                bulk supercell are also written to "{formula}_bulk/{subfolder}".
                (Default: False)
            **kwargs:
                Keyword arguments to pass to `DefectDictSet.write_input()`.
        """
        if self.vasp_std is None:  # warns user if vasp_std is None
            return

        if defect_dir is None:
            defect_dir = self.defect_entry.name

        self._check_potcars_and_write_vasp_xxx_files(
            defect_dir,
            subfolder,
            unperturbed_poscar=unperturbed_poscar,
            vasp_xxx_attribute=self.vasp_std,
            **kwargs,
        )
        if bulk:
            bulk_supercell = self._check_bulk_supercell_and_warn()
            if bulk_supercell is None:
                return

            formula = bulk_supercell.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
            output_dir = os.path.dirname(defect_dir) if "/" in defect_dir else "."
            self._check_potcars_and_write_vasp_xxx_files(
                f"{output_dir}/{formula}_bulk",
                subfolder,
                unperturbed_poscar=True,
                vasp_xxx_attribute=self.bulk_vasp_std,
                **kwargs,
            )

    def write_nkred_std(
        self,
        defect_dir: Optional[str] = None,
        subfolder: Optional[str] = "vasp_nkred_std",
        unperturbed_poscar: bool = False,
        bulk: bool = False,
        **kwargs,
    ):
        """
        Write the input files for defect calculations using `vasp_std` (i.e.
        with a non-Γ-only kpoint mesh) and `NKRED(X,Y,Z)` INCAR tag(s) to
        downsample kpoints for the HF exchange part of hybrid DFT calculations,
        following the doped recommended defect calculation workflow (see docs).
        By default, sets `NKRED(X,Y,Z)` to 2 or 3 in the directions for which
        the k-point grid is divisible by this factor.

        By default, does not generate `POSCAR` (input structure) files, as
        these should be taken from the `CONTCAR`s of `ShakeNBreak` calculations
        (via `snb-groundstate`) or, if not following the recommended
        structure-searching workflow, from the `CONTCAR`s of `vasp_gam`
        calculations. If unperturbed `POSCAR` files are desired, set
        `unperturbed_poscar=True`.
        If bulk is True, the input files for a singlepoint calculation of the
        bulk supercell are also written to "{formula}_bulk/{subfolder}".

        Returns None and a warning if the input kpoint settings correspond to
        a Γ-only kpoint mesh (in which case `vasp_gam` should be used) or for
        GGA calculations (if `LHFCALC` is set to `False` in user_incar_settings,
        in which case `vasp_std` should be used).

        See the `RelaxSet.yaml` and `DefectSet.yaml` files in the
        `doped/VASP_sets` folder for the default `INCAR` and `KPOINT` settings,
        and `PotcarSet.yaml` for the default `POTCAR` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default `INCAR`/`POTCAR` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.

        The `DefectEntry` object is also written to a `json` file in
        `defect_dir` to aid calculation provenance.

        Args:
            defect_dir (str):
                Folder in which to create the VASP defect calculation inputs.
                Default is to use the DefectEntry name (e.g. "Y_i_C4v_O1.92_+2"
                etc.), from `self.defect_entry.name`. If this attribute is not
                set, it is automatically generated according to the doped
                convention (using `get_defect_name_from_entry()`).
            subfolder (str):
                Output folder structure is `<defect_dir>/<subfolder>` where
                `subfolder` = 'vasp_nkred_std' by default. Setting `subfolder`
                to `None` will write the `vasp_nkred_std` input files directly
                to the `<defect_dir>` folder, with no subfolders created.
            unperturbed_poscar (bool):
                If True, write the unperturbed defect POSCARs to the generated
                folders as well. Not recommended, as the recommended workflow is
                to initially perform `vasp_gam` ground-state structure searching
                using ShakeNBreak (https://shakenbreak.readthedocs.io), then
                continue the `vasp_std` relaxations from the 'Groundstate` `CONTCAR`s.
                (default: False)
            bulk (bool):
                If True, the input files for a singlepoint calculation of the
                bulk supercell are also written to "{formula}_bulk/{subfolder}".
                (Default: False)
            **kwargs:
                Keyword arguments to pass to `DefectDictSet.write_input()`.
        """
        if self.vasp_nkred_std is None:  # warns user if vasp_nkred_std is None
            return

        if defect_dir is None:
            defect_dir = self.defect_entry.name

        self._check_potcars_and_write_vasp_xxx_files(
            defect_dir,
            subfolder,
            unperturbed_poscar=unperturbed_poscar,
            vasp_xxx_attribute=self.vasp_nkred_std,
            **kwargs,
        )
        if bulk:
            bulk_supercell = self._check_bulk_supercell_and_warn()
            if bulk_supercell is None:
                return

            formula = bulk_supercell.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
            output_dir = os.path.dirname(defect_dir) if "/" in defect_dir else "."
            self._check_potcars_and_write_vasp_xxx_files(
                f"{output_dir}/{formula}_bulk",
                subfolder,
                unperturbed_poscar=True,
                vasp_xxx_attribute=self.bulk_vasp_nkred_std,
                **kwargs,
            )

    def write_ncl(
        self,
        defect_dir: Optional[str] = None,
        subfolder: Optional[str] = "vasp_ncl",
        unperturbed_poscar: bool = False,
        bulk: bool = False,
        **kwargs,
    ):
        """
        Write the input files for VASP defect supercell singlepoint
        calculations with spin-orbit coupling (SOC) included (LSORBIT = True),
        using `vasp_ncl`. By default, does not generate `POSCAR` (input
        structure) files, as these should be taken from the `CONTCAR`s of
        `vasp_std` relaxations (originally from `ShakeNBreak` structure-
        searching relaxations), or directly from `ShakeNBreak` calculations
        (via `snb-groundstate`) if only Γ-point reciprocal space sampling is
        required. If unperturbed `POSCAR` files are desired, set
        `unperturbed_poscar=True`.

        If `DefectRelaxSet.soc` is False, then this returns None and a warning.
        If the `soc` parameter is not set when initializing `DefectRelaxSet`,
        then it is set to True for systems with a max atomic number (Z) >= 31
        (i.e. further down the periodic table than Zn), otherwise False.
        If bulk is True, the input files for a singlepoint calculation of the
        bulk supercell are also written to "{formula}_bulk/{subfolder}".

        See the `RelaxSet.yaml` and `DefectSet.yaml` files in the
        `doped/VASP_sets` folder for the default `INCAR` and `KPOINT` settings,
        and `PotcarSet.yaml` for the default `POTCAR` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default `INCAR`/`POTCAR` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.

        The `DefectEntry` object is also written to a `json` file in
        `defect_dir` to aid calculation provenance.

        Args:
            defect_dir (str):
                Folder in which to create the VASP defect calculation inputs.
                Default is to use the DefectEntry name (e.g. "Y_i_C4v_O1.92_+2"
                etc.), from `self.defect_entry.name`. If this attribute is not
                set, it is automatically generated according to the doped
                convention (using `get_defect_name_from_entry()`).
            subfolder (str):
                Output folder structure is `<defect_dir>/<subfolder>` where
                `subfolder` = 'vasp_ncl' by default. Setting `subfolder` to
                `None` will write the `vasp_ncl` input files directly to the
                `<defect_dir>` folder, with no subfolders created.
            unperturbed_poscar (bool):
                If True, write the unperturbed defect POSCARs to the generated
                folders as well. Not recommended, as the recommended workflow is
                to initially perform `vasp_gam` ground-state structure searching
                using ShakeNBreak (https://shakenbreak.readthedocs.io), then
                continue the `vasp_std` relaxations from the 'Groundstate'
                `CONTCAR`s (first with NKRED if using hybrid DFT, then without),
                then use the `vasp_std` `CONTCAR`s as the input structures for
                the final `vasp_ncl` singlepoint calculations.
                (default: False)
            bulk (bool):
                If True, the input files for a singlepoint calculation of the
                bulk supercell are also written to "{formula}_bulk/{subfolder}".
                (Default: False)
            **kwargs:
                Keyword arguments to pass to `DefectDictSet.write_input()`.
        """
        if self.vasp_ncl is None:
            return

        if defect_dir is None:
            defect_dir = self.defect_entry.name

        self._check_potcars_and_write_vasp_xxx_files(
            defect_dir,
            subfolder,
            unperturbed_poscar=unperturbed_poscar,
            vasp_xxx_attribute=self.vasp_ncl,
            **kwargs,
        )
        if bulk:
            bulk_supercell = self._check_bulk_supercell_and_warn()
            if bulk_supercell is None:
                return

            formula = bulk_supercell.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
            # get output dir: (folder above defect_dir if defect_dir is a subfolder)
            output_dir = os.path.dirname(defect_dir) if "/" in defect_dir else "."
            self._check_potcars_and_write_vasp_xxx_files(
                f"{output_dir}/{formula}_bulk",
                subfolder,
                unperturbed_poscar=True,
                vasp_xxx_attribute=self.bulk_vasp_ncl,
                **kwargs,
            )

    def write_all(
        self,
        defect_dir: Optional[str] = None,
        unperturbed_poscar: bool = False,
        vasp_gam: bool = False,
        bulk: Union[bool, str] = False,
        **kwargs,
    ):
        """
        Write all VASP input files to subfolders in the `defect_dir` folder.

        The following subfolders are generated:
        - vasp_nkred_std -> Defect relaxation with a kpoint mesh and using `NKRED`.
            Not generated for GGA calculations (if `LHFCALC` is set to `False` in
            user_incar_settings) or if only Γ-point sampling required.
        - vasp_std -> Defect relaxation with a kpoint mesh, not using `NKRED`. Not
            generated if only Γ-point sampling required.
        - vasp_ncl -> Singlepoint (static) energy calculation with SOC included.
            Generated if `soc=True`. If `soc` is not set, then by default is only
            generated for systems with a max atomic number (Z) >= 31 (i.e. further
            down the periodic table than Zn).

        If vasp_gam=True (not recommended) or self.vasp_std = None (i.e. Γ-only
        _k_-point sampling converged for the kpoints settings used), then outputs:
        - vasp_gam -> Γ-point only defect relaxation. Not needed if ShakeNBreak
            structure searching has been performed (recommended).

        By default, does not generate a `vasp_gam` folder unless `self.vasp_std`
        is None (i.e. only Γ-point sampling required for this system), as
        `vasp_gam` calculations should be performed using `ShakeNBreak` for
        defect structure-searching and initial relaxations. If `vasp_gam` files
        are desired, set `vasp_gam=True`.

        By default, `POSCAR` files are not generated for the `vasp_(nkred_)std`
        (and `vasp_ncl` if `self.soc` is True) folders, as these should
        be taken from `ShakeNBreak` calculations (via `snb-groundstate`)
        or, if not following the recommended structure-searching workflow,
        from the `CONTCAR`s of `vasp_gam` calculations. If including SOC
        effects (`self.soc = True`), then the `vasp_std` `CONTCAR`s
        should be used as the `vasp_ncl` `POSCAR`s. If unperturbed
        `POSCAR` files are desired for the `vasp_(nkred_)std` (and `vasp_ncl`)
        folders, set `unperturbed_poscar=True`.

        Input files for the singlepoint (static) bulk supercell reference
        calculation are also written to "{formula}_bulk/{subfolder}" if `bulk`
        is True (False by default), where `subfolder` corresponds to the final
        (highest accuracy) VASP calculation in the workflow (i.e. `vasp_ncl` if
        `self.soc=True`, otherwise `vasp_std` or `vasp_gam` if only Γ-point
        reciprocal space sampling is required). If `bulk = "all"`, then the
        input files for all VASP calculations in the workflow are written to
        the bulk supercell folder.

        See the `RelaxSet.yaml` and `DefectSet.yaml` files in the
        `doped/VASP_sets` folder for the default `INCAR` and `KPOINT` settings,
        and `PotcarSet.yaml` for the default `POTCAR` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default `INCAR`/`POTCAR` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.

        The `DefectEntry` object is also written to a `json` file in
        `defect_dir` to aid calculation provenance.

        Args:
            defect_dir (str):
                Folder in which to create the VASP defect calculation inputs.
                Default is to use the DefectEntry name (e.g. "Y_i_C4v_O1.92_+2"
                etc.), from `self.defect_entry.name`. If this attribute is not
                set, it is automatically generated according to the doped
                convention (using `get_defect_name_from_entry()`).
                Output folder structure is `<defect_dir>/<subfolder>` where
                `subfolder` is the name of the corresponding VASP program to run
                (e.g. `vasp_std`).
            unperturbed_poscar (bool):
                If True, write the unperturbed defect POSCARs to the generated
                folders as well. Not recommended, as the recommended workflow is
                to initially perform `vasp_gam` ground-state structure searching
                using ShakeNBreak (https://shakenbreak.readthedocs.io), then
                continue the `vasp_std` relaxations from the 'Groundstate'
                `CONTCAR`s (first with NKRED if using hybrid DFT, then without),
                then use the `vasp_std` `CONTCAR`s as the input structures for
                the final `vasp_ncl` singlepoint calculations.
                (default: False)
            vasp_gam (bool):
                If True, write the `vasp_gam` input files, with unperturbed defect
                POSCAR. Not recommended, as the recommended workflow is to initially
                perform `vasp_gam` ground-state structure searching using ShakeNBreak
                (https://shakenbreak.readthedocs.io), then continue the `vasp_std`
                relaxations from the 'Groundstate' `CONTCAR`s (first with NKRED if
                using hybrid DFT, then without), then if including SOC effects, use
                the `vasp_std` `CONTCAR`s as the input structures for the final
                `vasp_ncl` singlepoint calculations.
                (default: False)
            bulk (bool, str):
                If True, the input files for a singlepoint calculation of the
                bulk supercell are also written to "{formula}_bulk/{subfolder}",
                where `subfolder` corresponds to the final (highest accuracy)
                VASP calculation in the workflow (i.e. `vasp_ncl` if `self.soc=True`,
                otherwise `vasp_std` or `vasp_gam` if only Γ-point reciprocal space
                sampling is required). If `bulk = "all"` then the input files for
                all VASP calculations in the workflow (`vasp_gam`, `vasp_nkred_std`,
                `vasp_std`, `vasp_ncl` (if applicable)) are written to the bulk
                supercell folder.
                (Default: False)
            **kwargs:
                Keyword arguments to pass to `DefectDictSet.write_input()`.
        """
        # check `bulk` input:
        bulk_vasp = []
        if bulk and isinstance(bulk, str) and bulk.lower() == "all":
            bulk_vasp = ["vasp_gam", "vasp_nkred_std", "vasp_std", "vasp_ncl"]
        elif bulk and not isinstance(bulk, str) and bulk is not True:
            raise ValueError("Unrecognised input for `bulk` argument. Must be True, False, or 'all'.")

        # check if vasp_std is None, and if so, set vasp_gam to True:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if bulk is True:
                # get top-level VASP (ncl > std > gam): first self.vasp_xxx attribute that isn't None:
                top_vasp = next(  # top boy
                    (
                        vasp
                        for vasp in ["vasp_ncl", "vasp_std", "vasp_gam"]
                        if getattr(self, vasp) is not None
                    ),
                    None,
                )
                if top_vasp is None:
                    raise ValueError("No VASP input files to generate for the bulk supercell.")

                bulk_vasp = [top_vasp]

            if vasp_gam or self.vasp_std is None:
                self.write_gam(
                    defect_dir=defect_dir,
                    bulk=any("vasp_gam" in vasp_type for vasp_type in bulk_vasp),
                    **kwargs,
                )

            if self.vasp_std is not None:  # k-point mesh
                self.write_std(
                    defect_dir=defect_dir,
                    unperturbed_poscar=unperturbed_poscar,
                    bulk=any("vasp_std" in vasp_type for vasp_type in bulk_vasp),
                    **kwargs,
                )

            if self.vasp_nkred_std is not None:  # k-point mesh and not GGA
                self.write_nkred_std(
                    defect_dir=defect_dir,
                    unperturbed_poscar=unperturbed_poscar,
                    bulk=any("vasp_nkred_std" in vasp_type for vasp_type in bulk_vasp),
                    **kwargs,
                )

        if self.soc:  # SOC
            self.write_ncl(
                defect_dir=defect_dir,
                unperturbed_poscar=unperturbed_poscar,
                bulk=any("vasp_ncl" in vasp_type for vasp_type in bulk_vasp),
                **kwargs,
            )


class DefectsSet(MSONable):
    """
    An object for generating input files for VASP defect calculations from
    doped/pymatgen `DefectEntry` objects.
    """

    def __init__(
        self,
        defect_entries: Union[DefectsGenerator, Dict[str, DefectEntry], List[DefectEntry], DefectEntry],
        soc: Optional[bool] = None,
        user_incar_settings: Optional[dict] = None,
        user_kpoints_settings: Optional[dict] = None,
        user_potcar_functional: Optional[UserPotcarFunctional] = "PBE",
        user_potcar_settings: Optional[dict] = None,
        **kwargs,  # to allow POTCAR testing on GH Actions
    ):
        """
        Creates a dictionary of: {defect_species: DefectRelaxSet}.

        DefectRelaxSet has the attributes:
        - `DefectRelaxSet.vasp_gam` -> `DefectDictSet` for Gamma-point only
            relaxation. Not needed if ShakeNBreak structure searching has been
            performed (recommended), unless only Γ-point _k_-point sampling is
            required (converged) for your system, and no vasp_std calculations with
            multiple _k_-points are required (determined from kpoints settings).
        - `DefectRelaxSet.vasp_nkred_std` -> `DefectDictSet` for relaxation with a
            kpoint mesh and using `NKRED`. Not generated for GGA calculations (if
            `LHFCALC` is set to `False` in user_incar_settings) or if only Gamma
            kpoint sampling is required.
        - `DefectRelaxSet.vasp_std` -> `DefectDictSet` for relaxation with a kpoint
            mesh, not using `NKRED`. Not generated if only Gamma kpoint sampling is
            required.
        - `DefectRelaxSet.vasp_ncl` -> `DefectDictSet` for singlepoint (static)
            energy calculation with SOC included. Generated if `soc=True`. If `soc`
            is not set, then by default is only generated for systems with a max
            atomic number (Z) >= 31 (i.e. further down the periodic table than Zn).
        where `DefectDictSet` is an extension of `pymatgen`'s `DictSet` class for
        defect calculations, with `incar`, `poscar`, `kpoints` and `potcar`
        attributes for the corresponding VASP defect calculations (see docstring).

        See the `RelaxSet.yaml` and `DefectSet.yaml` files in the
        `doped/VASP_sets` folder for the default `INCAR` settings, and
        `PotcarSet.yaml` for the default `POTCAR` settings.

        Note that any changes to the default `INCAR`/`POTCAR` settings should
        be consistent with those used for all defect and competing phase (chemical
        potential) calculations.

        Args:
            defect_entries (`DefectsGenerator` or dict/list of `DefectEntry`s, or single `DefectEntry`):
                Either a `DefectsGenerator` object, or a dictionary/list of
                `DefectEntry`s, or a single `DefectEntry` object, for which
                to generate VASP input files.
                If a `DefectsGenerator` object or a dictionary (->
                {defect_species: DefectEntry}), the defect folder names will be
                set equal to `defect_species`. If a list or single `DefectEntry`
                object is provided, the defect folder names will be set equal to
                `DefectEntry.name` if the `name` attribute is set, otherwise
                generated according to the `doped` convention (see doped.generation).
                Defect charge states are taken from `DefectEntry.charge_state`.
            soc (bool):
                Whether to generate `vasp_ncl` DefectDictSet attribute for spin-orbit
                coupling singlepoint (static) energy calculations. If not set, then
                by default is set to True if the max atomic number (Z) in the
                structure is >= 31 (i.e. further down the periodic table than Zn).
            user_incar_settings (dict):
                Dictionary of user INCAR settings (AEXX, NCORE etc.) to override
                default settings. Highly recommended to look at output INCARs or the
                `RelaxSet.yaml` and `DefectSet.yaml` files in the `doped/VASP_sets`
                folder, to see what the default INCAR settings are. Note that any
                flags that aren't numbers or True/False need to be input as strings
                with quotation marks (e.g. `{"ALGO": "All"}`).
                (default: None)
            user_kpoints_settings (dict or Kpoints):
                Dictionary of user KPOINTS settings (in pymatgen DictSet() format)
                e.g. {"reciprocal_density": 123}, or a Kpoints object, to use for the
                `vasp_std`, `vasp_nkred_std` and `vasp_ncl` DefectDictSets (Γ-only for
                `vasp_gam`). Default is Gamma-centred, reciprocal_density = 100 [Å⁻³].
            user_potcar_functional (str):
                POTCAR functional to use. Default is "PBE" and if this fails, tries
                "PBE_52", then "PBE_54".
            user_potcar_settings (dict):
                Override the default POTCARs, e.g. {"Li": "Li_sv"}. See
                `doped/VASP_setsPotcarSet.yaml` for the default `POTCAR` set.
            **kwargs: Additional kwargs to pass to DictSet().

        Attributes:
            defect_sets (Dict):
                Dictionary of {defect_species: `DefectRelaxSet`}.
            defect_entries (Dict):
                Dictionary of {defect_species: DefectEntry} for the input defect
                species, for which to generate VASP input files.
            json_obj (Union[Dict, DefectsGenerator]):
                Either the DefectsGenerator object if input `defect_entries` is a
                `DefectsGenerator` object, otherwise the `defect_entries` dictionary,
                which will be written to file when `write_files()` is called, to
                aid calculation provenance.
            json_name (str):
                Name of the JSON file to save the `json_obj` to.

            Input parameters are also set as attributes.
        """
        self.user_incar_settings = user_incar_settings
        self.user_kpoints_settings = user_kpoints_settings
        self.user_potcar_functional = user_potcar_functional
        self.user_potcar_settings = user_potcar_settings
        self.kwargs = kwargs
        self.defect_entries, self.json_name, self.json_obj = self._format_defect_entries_input(
            defect_entries
        )

        if soc is not None:
            self.soc = soc
        else:

            def _get_atomic_numbers(defect_entry):
                # use defect supercell rather than defect.defect_structure because could be e.g. a
                # vacancy in a 2-atom primitive structure where the atom being removed is the heavy
                # (Z>=31) one
                if defect_entry.defect_supercell is not None:
                    return defect_entry.defect_supercell.atomic_numbers
                if defect_entry.sc_entry.structure is not None:
                    return defect_entry.sc_entry.structure.atomic_numbers

                raise ValueError(
                    "Defect supercell needs to be defined in the DefectEntry attributes, but both "
                    "DefectEntry.defect_supercell and DefectEntry.sc_entry.structure are None!"
                )

            max_atomic_num = np.max(
                [
                    np.max(_get_atomic_numbers(defect_entry))
                    for defect_entry in self.defect_entries.values()
                ]
            )
            self.soc = max_atomic_num >= 31

        self.defect_sets: Dict[str, DefectRelaxSet] = {}

        for defect_species, defect_entry in self.defect_entries.items():
            self.defect_sets[defect_species] = DefectRelaxSet(
                defect_entry=defect_entry,
                charge_state=defect_entry.charge_state,
                soc=self.soc,
                user_incar_settings=self.user_incar_settings,
                user_kpoints_settings=self.user_kpoints_settings,
                user_potcar_functional=self.user_potcar_functional,
                user_potcar_settings=self.user_potcar_settings,
                **self.kwargs,
            )

    def _format_defect_entries_input(
        self,
        defect_entries: Union[DefectsGenerator, Dict[str, DefectEntry], List[DefectEntry], DefectEntry],
    ) -> Tuple[Dict[str, DefectEntry], str, Union[Dict[str, DefectEntry], DefectsGenerator]]:
        """
        Helper function to format input `defect_entries` into a named
        dictionary of `DefectEntry` objects. Also returns the name of the JSON
        file and object to serialise when writing the VASP input to files. This
        is the DefectsGenerator object if `defect_entries` is a
        `DefectsGenerator` object, otherwise the dictionary of `DefectEntry`
        objects.

        Args:
            defect_entries (`DefectsGenerator` or dict/list of `DefectEntry`s, or single `DefectEntry`):
                Either a `DefectsGenerator` object, or a dictionary/list of
                `DefectEntry`s, or a single `DefectEntry` object, for which
                to generate VASP input files.
                If a `DefectsGenerator` object or a dictionary (->
                {defect_species: DefectEntry}), the defect folder names will be
                set equal to `defect_species`. If a list or single `DefectEntry`
                object is provided, the defect folder names will be set equal to
                `DefectEntry.name` if the `name` attribute is set, otherwise
                generated according to the `doped` convention (see doped.generation).
        """
        json_filename = "defect_entries.json"  # global statement in case, but should be skipped
        json_obj = defect_entries
        if type(defect_entries).__name__ == "DefectsGenerator":
            defect_entries = cast(DefectsGenerator, defect_entries)
            formula = defect_entries.primitive_structure.composition.get_reduced_formula_and_factor(
                iupac_ordering=True
            )[0]
            json_filename = f"{formula}_defects_generator.json"
            json_obj = defect_entries
            defect_entries = defect_entries.defect_entries

        elif isinstance(defect_entries, DefectEntry):
            defect_entries = [defect_entries]
        if isinstance(
            defect_entries, list
        ):  # also catches case where defect_entries is a single DefectEntry, from converting to list above
            # need to convert to dict with doped names as keys:
            defect_entry_list = copy.deepcopy(defect_entries)
            with contextlib.suppress(AttributeError):  # sort by conventional cell fractional
                # coordinates if these are defined, to aid deterministic naming
                defect_entry_list.sort(key=lambda x: _frac_coords_sort_func(x.conv_cell_frac_coords))

            # figure out which DefectEntry objects need to be named (don't name if already named)
            defect_entries_to_name = [
                defect_entry for defect_entry in defect_entry_list if not hasattr(defect_entry, "name")
            ]
            new_named_defect_entries_dict = name_defect_entries(defect_entries_to_name)
            # set name attribute: (these are names without charges!)
            for defect_name_wout_charge, defect_entry in new_named_defect_entries_dict.items():
                defect_entry.name = (
                    f"{defect_name_wout_charge}_{'+' if defect_entry.charge_state > 0 else ''}"
                    f"{defect_entry.charge_state}"
                )

            # if any duplicate names, crash (and burn, b...)
            if len({defect_entry.name for defect_entry in defect_entry_list}) != len(defect_entry_list):
                raise ValueError(
                    "Some defect entries have the same name, due to mixing of named and unnamed input "
                    "`DefectEntry`s! This would cause defect folders to be overwritten. Please check "
                    "your DefectEntry names and/or generate your defects using DefectsGenerator instead."
                )

            defect_entries = {defect_entry.name: defect_entry for defect_entry in defect_entry_list}
            formula = defect_entry_list[0].defect.structure.composition.get_reduced_formula_and_factor(
                iupac_ordering=True
            )[0]
            json_filename = f"{formula}_defect_entries.json"
            json_obj = defect_entries

        # check correct format:
        if isinstance(defect_entries, dict) and not all(
            isinstance(defect_entry, DefectEntry) for defect_entry in defect_entries.values()
        ):
            raise TypeError(
                f"Input defect_entries dict must be of the form {{defect_name: DefectEntry}}, got dict "
                f"with values of type {[type(value) for value in defect_entries.values()]} instead"
            )

        if not isinstance(defect_entries, dict):
            raise TypeError(
                f"Input defect_entries must be of type DefectsGenerator, dict, list or DefectEntry, got "
                f"type {type(defect_entries)} instead."
            )

        return defect_entries, json_filename, json_obj  # type: ignore

    @staticmethod
    def _write_defect(args):
        defect_species, defect_entry, output_dir, unperturbed_poscar, vasp_gam, bulk, kwargs = args
        defect_dir = os.path.join(output_dir, defect_species)
        defect_entry.write_all(
            defect_dir=defect_dir,
            unperturbed_poscar=unperturbed_poscar,
            vasp_gam=vasp_gam,
            bulk=bulk,
            **kwargs,
        )

    def write_files(
        self,
        output_dir: str = ".",
        unperturbed_poscar: bool = False,
        vasp_gam: bool = False,
        bulk: Union[bool, str] = True,
        processes: Optional[int] = None,
        **kwargs,
    ):
        """
        Write VASP input files to folders for all defects in
        `self.defect_entries`. Folder names are set to the key of the
        DefectRelaxSet in `self.defect_sets` (same as self.defect_entries keys,
        see `DefectsSet` docstring).

        For each defect folder, the following subfolders are generated:
        - vasp_nkred_std -> Defect relaxation with a kpoint mesh and using `NKRED`.
            Not generated for GGA calculations (if `LHFCALC` is set to `False` in
            user_incar_settings) or if only Γ-point sampling required.
        - vasp_std -> Defect relaxation with a kpoint mesh, not using `NKRED`. Not
            generated if only Γ-point sampling required.
        - vasp_ncl -> Singlepoint (static) energy calculation with SOC included.
            Generated if `soc=True`. If `soc` is not set, then by default is only
            generated for systems with a max atomic number (Z) >= 31 (i.e. further
            down the periodic table than Zn).

        If vasp_gam=True (not recommended) or self.vasp_std = None (i.e. Γ-only
        _k_-point sampling converged for the kpoints settings used), then outputs:
        - vasp_gam -> Γ-point only defect relaxation. Not needed if ShakeNBreak
            structure searching has been performed (recommended).

        By default, does not generate a `vasp_gam` folder unless
        `DefectRelaxSet.vasp_std` is None (i.e. only Γ-point sampling required
        for this system), as `vasp_gam` calculations should be performed using
        `ShakeNBreak` for defect structure-searching and initial relaxations.
        If `vasp_gam` files are desired, set `vasp_gam=True`.

        By default, `POSCAR` files are not generated for the `vasp_(nkred_)std`
        (and `vasp_ncl` if `self.soc` is True) folders, as these should
        be taken from `ShakeNBreak` calculations (via `snb-groundstate`)
        or, if not following the recommended structure-searching workflow,
        from the `CONTCAR`s of `vasp_gam` calculations. If including SOC
        effects (`self.soc = True`), then the `vasp_std` `CONTCAR`s
        should be used as the `vasp_ncl` `POSCAR`s. If unperturbed
        `POSCAR` files are desired for the `vasp_(nkred_)std` (and `vasp_ncl`)
        folders, set `unperturbed_poscar=True`.

        Input files for the singlepoint (static) bulk supercell reference
        calculation are also written to "{formula}_bulk/{subfolder}" if `bulk`
        is True (default), where `subfolder` corresponds to the final (highest
        accuracy) VASP calculation in the workflow (i.e. `vasp_ncl` if
        `self.soc=True`, otherwise `vasp_std` or `vasp_gam` if only Γ-point
        reciprocal space sampling is required). If `bulk = "all"`, then the
        input files for all VASP calculations in the workflow are written to
        the bulk supercell folder, or if `bulk = False`, then no bulk folder
        is created.

        The `DefectEntry` objects are also written to `json` files in the defect
        folders, to aid calculation provenance.

        See the `RelaxSet.yaml` and `DefectSet.yaml` files in the
        `doped/VASP_sets` folder for the default `INCAR` and `KPOINT` settings,
        and `PotcarSet.yaml` for the default `POTCAR` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default `INCAR`/`POTCAR` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.

        Args:
            output_dir (str):
                Folder in which to create the VASP defect calculation folders.
                Default is the current directory ("."). Output folder structure
                is `<output_dir>/<defect_species>/<subfolder>` where
                `defect_species` is the key of the DefectRelaxSet in
                `self.defect_sets` (same as `self.defect_entries` keys, see
                `DefectsSet` docstring) and `subfolder` is the name of the
                corresponding VASP program to run (e.g. `vasp_std`).
            unperturbed_poscar (bool):
                If True, write the unperturbed defect POSCARs to the generated
                folders as well. Not recommended, as the recommended workflow is
                to initially perform `vasp_gam` ground-state structure searching
                using ShakeNBreak (https://shakenbreak.readthedocs.io), then
                continue the `vasp_std` relaxations from the 'Groundstate'
                `CONTCAR`s (first with NKRED if using hybrid DFT, then without),
                then use the `vasp_std` `CONTCAR`s as the input structures for
                the final `vasp_ncl` singlepoint calculations.
                (default: False)
            vasp_gam (bool):
                If True, write the `vasp_gam` input files, with unperturbed defect
                POSCARs. Not recommended, as the recommended workflow is to initially
                perform `vasp_gam` ground-state structure searching using ShakeNBreak
                (https://shakenbreak.readthedocs.io), then continue the `vasp_std`
                relaxations from the 'Groundstate' `CONTCAR`s (first with NKRED if
                using hybrid DFT, then without), then if including SOC effects, use
                the `vasp_std` `CONTCAR`s as the input structures for the final
                `vasp_ncl` singlepoint calculations.
                (default: False)
            bulk (bool, str):
                If True, the input files for a singlepoint calculation of the
                bulk supercell are also written to "{formula}_bulk/{subfolder}",
                where `subfolder` corresponds to the final (highest accuracy)
                VASP calculation in the workflow (i.e. `vasp_ncl` if `self.soc=True`,
                otherwise `vasp_std` or `vasp_gam` if only Γ-point reciprocal space
                sampling is required). If `bulk = "all"` then the input files for
                all VASP calculations in the workflow (`vasp_gam`, `vasp_nkred_std`,
                `vasp_std`, `vasp_ncl` (if applicable)) are written to the bulk
                supercell folder.
                (Default: False)
            processes (int):
                Number of processes to use for multiprocessing for file writing.
                If not set, defaults to one less than the number of CPUs available.
            **kwargs:
                Keyword arguments to pass to `DefectDictSet.write_input()`.
        """
        args_list = [
            (
                defect_species,
                defect_entry,
                output_dir,
                unperturbed_poscar,
                vasp_gam,
                bulk if i == 0 else False,  # only write bulk folder(s) for first defect
                kwargs,
            )
            for i, (defect_species, defect_entry) in enumerate(self.defect_sets.items())
        ]
        with Pool(processes=processes or cpu_count() - 1) as pool:
            for _ in tqdm(
                pool.imap(self._write_defect, args_list),
                total=len(args_list),
                desc="Generating and writing input files",
            ):
                pass

        dumpfn(self.json_obj, self.json_name)


# TODO: Remove these functions once confirmed all functionality is in `chemical_potentials.py`;
# need `vasp_ncl_chempot` generation, `vaspup2.0` `input` folder with `CONFIG` generation as an
# option, improve chemical_potentials docstrings (i.e. mention defaults, note in notebooks if changing
# `INCAR`/`POTCAR` settings for competing phase production calcs, should also do with defect
# supercell calcs (and note this in vasp_input as well)), ensure consistent INCAR tags in defect
# supercell defaults and competing phase defaults, point to DefectSet in docstrings for defaults
# (noting the other INCAR tags that are changed).
# def _vasp_converge_files(
#     structure: "pymatgen.core.Structure",
#     input_dir: Optional[str] = None,
#     incar_settings: Optional[dict] = None,
#     potcar_settings: Optional[dict] = None,
#     config: Optional[str] = None,
# ) -> None:
#     """
#     Generates input files for single-shot GGA convergence test calculations.
#
#     Automatically sets ISMEAR (in INCAR) to 2 (if metallic) or 0 if not.
#     Recommended to use with vaspup2.0
#     Args:
#         structure (Structure object):
#             Structure to create input files for.
#         input_dir (str):
#             Folder in which to create 'input' folder with VASP input files.
#             (default: None)
#         incar_settings (dict):
#             Dictionary of user INCAR settings (AEXX, NCORE etc.) to override default settings.
#             Highly recommended to look at output INCARs or doped.vasp_input
#             source code, to see what the default INCAR settings are. Note that any flags that
#             aren't numbers or True/False need to be input as strings with quotation marks
#             (e.g. `{"ALGO": "All"}`).
#             (default: None)
#         config (str):
#             CONFIG file string. If provided, will also write the CONFIG file (to automate
#             convergence tests with vaspup2.0) to each 'input' directory.
#             (default: None)
#         potcar_settings (dict):
#             Dictionary of user POTCAR settings to override default settings.
#             Highly recommended to look at `default_potcar_dict` from doped.vasp_input to see what
#             the (Pymatgen) syntax and doped default settings are.
#             (default: None).
#     """
#     # Variable parameters first
#     vaspconvergeincardict = {
#         "# May need to change ISMEAR, NCORE, KPAR, AEXX, ENCUT, NUPDOWN, "
#         + "ISPIN": "variable parameters",
#         "NUPDOWN": "0 # But could be >0 if magnetic behaviour present",
#         "NCORE": 12,
#         "#KPAR": 1,
#         "ENCUT": 400,
#         "ISMEAR": "0 # Non-metal, use Gaussian smearing",
#         "ISPIN": "1 # Change to 2 if spin polarisation or magnetic behaviour present",
#         "GGA": "PS",  # PBEsol
#         "ALGO": "Normal # Change to All if ZHEGV, FEXCP/F or ZBRENT errors encountered",
#         "EDIFF": 1e-06,
#         "EDIFFG": -0.01,
#         "IBRION": -1,
#         "ISIF": 3,
#         "LASPH": True,
#         "LORBIT": 14,
#         "LREAL": False,
#         "LWAVE": "False # Save filespace, shouldn't need WAVECAR from convergence tests",
#         "NEDOS": 2000,
#         "NELM": 100,
#         "NSW": 0,
#         "PREC": "Accurate",
#         "SIGMA": 0.2,
#     }
#     if all(is_metal(element) for element in structure.composition.elements):
#         vaspconvergeincardict["ISMEAR"] = "2 # Metal, use Methfessel-Paxton smearing scheme"
#     if incar_settings:
#         for k in incar_settings:  # check INCAR flags and warn if they don't exist (
#             # typos)
#             if k not in incar_params.keys():  # this code is taken from pymatgen.io.vasp.inputs
#                 warnings.warn(  # but only checking keys, not values so we can add comments etc
#                     "Cannot find %s from your user_incar_settings in the list of INCAR flags" % (k),
#                     BadIncarWarning,
#                 )
#         vaspconvergeincardict.update(incar_settings)
#
#     # Directory
#     vaspconvergeinputdir = input_dir + "/input/" if input_dir else "VASP_Files/input/"
#     if not os.path.exists(vaspconvergeinputdir):
#         os.makedirs(vaspconvergeinputdir)
#
#     # POTCAR
#     potcar_dict = deepcopy(default_potcar_dict)
#     if potcar_settings:
#         if "POTCAR_FUNCTIONAL" in potcar_settings:
#             potcar_dict["POTCAR_FUNCTIONAL"] = potcar_settings["POTCAR_FUNCTIONAL"]
#         if "POTCAR" in potcar_settings:
#             potcar_dict["POTCAR"].update(potcar_settings.pop("POTCAR"))
#     vaspconvergeinput = DictSet(structure, config_dict=potcar_dict)
#     vaspconvergeinput.potcar.write_file(vaspconvergeinputdir + "POTCAR")
#
#     vaspconvergekpts = Kpoints().from_dict(
#         {"comment": "Kpoints from vasp_gam_files", "generation_style": "Gamma"}
#     )
#     vaspconvergeincar = Incar.from_dict(vaspconvergeincardict)
#     vaspconvergeincar.write_file(vaspconvergeinputdir + "INCAR")
#
#     vaspconvergeposcar = Poscar(structure)
#     vaspconvergeposcar.write_file(vaspconvergeinputdir + "POSCAR")
#
#     vaspconvergekpts.write_file(vaspconvergeinputdir + "KPOINTS")
#     # generate CONFIG file
#     if config:
#         with open(vaspconvergeinputdir + "CONFIG", "w+") as config_file:
#             config_file.write(config)
#         with open(vaspconvergeinputdir + "CONFIG", "a") as config_file:
#             config_file.write(f"""\nname="{input_dir[13:]}" # input_dir""")
#
#
# # Input files for vasp_std
#
#
# def _vasp_std_chempot(
#     structure: "pymatgen.core.Structure",
#     input_dir: Optional[str] = None,
#     incar_settings: Optional[dict] = None,
#     kpoints_settings: Optional[dict] = None,
#     potcar_settings: Optional[dict] = None,
# ) -> None:
#     """
#     Generates POSCAR, INCAR, POTCAR and KPOINTS for vasp_std chemical
#     potentials relaxation.:
#
#     Args:
#         structure (Structure object):
#             Structure to create input files for.
#         input_dir (str):
#             Folder in which to create vasp_std calculation inputs folder
#             (default: None)
#         incar_settings (dict):
#             Dictionary of user INCAR settings (AEXX, NCORE etc.) to override default settings.
#             Highly recommended to look at output INCARs or doped.vasp_input
#             source code, to see what the default INCAR settings are. Note that any flags that
#             aren't numbers or True/False need to be input as strings with quotation marks
#             (e.g. `{"ALGO": "All"}`).
#             (default: None)
#         kpoints_settings (dict):
#             Dictionary of user KPOINTS settings (in pymatgen Kpoints.from_dict() format). Common
#             options would be "generation_style": "Monkhorst" (rather than "Gamma"),
#             and/or "kpoints": [[3, 3, 1]] etc.
#             Default KPOINTS is Gamma-centred 2 x 2 x 2 mesh.
#             (default: None)
#         potcar_settings (dict):
#             Dictionary of user POTCAR settings to override default settings.
#             Highly recommended to look at `default_potcar_dict` from doped.vasp_input to see what
#             the (Pymatgen) syntax and doped default settings are.
#             (default: None).
#     """
#     # INCAR Parameters
#     vaspstdincardict = {
#         "# May need to change NCORE, KPAR, ENCUT" + "ISPIN, POTIM": "variable parameters",
#         "NCORE": 12,
#         "KPAR": 2,
#         "AEXX": 0.25,
#         "ENCUT": 400,
#         "POTIM": 0.2,
#         "LSUBROT": "False # Change to True if relaxation poorly convergent",
#         "ICORELEVEL": "0 # Needed if using the Kumagai-Oba (eFNV) anisotropic charge correction",
#         "ALGO": "Normal # Change to All if ZHEGV, FEXCP/F or ZBRENT errors encountered",
#         "EDIFF": 1e-06,  # May need to reduce for tricky relaxations",
#         "EDIFFG": -0.01,
#         "HFSCREEN": 0.2,  # assuming HSE06
#         "IBRION": "1 # May need to change to 2 for difficult/poorly-convergent relaxations",
#         "ISIF": 3,
#         "ISMEAR": 0,
#         "LASPH": True,
#         "LHFCALC": True,
#         "LORBIT": 14,
#         "LREAL": False,
#         "LVHAR": "True # Needed if using the Freysoldt (FNV) charge correction scheme",
#         "LWAVE": True,
#         "NEDOS": 2000,
#         "NELM": 100,
#         "NSW": 200,
#         "PREC": "Accurate",
#         "PRECFOCK": "Fast",
#         "SIGMA": 0.05,
#     }
#
#     # Directory
#     vaspstdinputdir = input_dir + "/vasp_std/" if input_dir else "VASP_Files/vasp_std/"
#     if not os.path.exists(vaspstdinputdir):
#         os.makedirs(vaspstdinputdir)
#
#     # POTCAR
#     potcar_dict = default_potcar_dict
#     if potcar_settings:
#         if "POTCAR_FUNCTIONAL" in potcar_settings:
#             potcar_dict["POTCAR_FUNCTIONAL"] = potcar_settings["POTCAR_FUNCTIONAL"]
#         if "POTCAR" in potcar_settings:
#             potcar_dict["POTCAR"].update(potcar_settings.pop("POTCAR"))
#     vaspstdinput = DictSet(structure, config_dict=potcar_dict)
#     vaspstdinput.potcar.write_file(vaspstdinputdir + "POTCAR")
#
#     if all(is_metal(element) for element in structure.composition.elements):
#         vaspstdincardict["ISMEAR"] = "2 # Metal, use Methfessel-Paxton smearing scheme"
#     if all(is_metal(element) for element in structure.composition.elements):
#         vaspstdincardict["SIGMA"] = 0.02
#
#     if incar_settings:
#         for k in incar_settings:  # check INCAR flags and warn if they don't exist (typos)
#             if k not in incar_params.keys():  # this code is taken from pymatgen.io.vasp.inputs
#                 warnings.warn(  # but only checking keys, not values so we can add comments etc
#                     "Cannot find %s from your user_incar_settings in the list of INCAR flags" % (k),
#                     BadIncarWarning,
#                 )
#         vaspstdincardict.update(incar_settings)
#
#     # POSCAR
#     vaspstdposcar = Poscar(structure)
#     vaspstdposcar.write_file(vaspstdinputdir + "POSCAR")
#
#     # KPOINTS
#     vaspstdkpointsdict = {
#         "comment": "Kpoints from doped.vasp_std_files",
#         "generation_style": "Gamma",  # Set to Monkhorst for Monkhorst-Pack generation
#         "kpoints": [[2, 2, 2]],
#     }
#     if kpoints_settings:
#         vaspstdkpointsdict.update(kpoints_settings)
#     vaspstdkpts = Kpoints.from_dict(vaspstdkpointsdict)
#     vaspstdkpts.write_file(vaspstdinputdir + "KPOINTS")
#
#     # INCAR
#     vaspstdincar = Incar.from_dict(vaspstdincardict)
#     with zopen(vaspstdinputdir + "INCAR", "wt") as incar_file:
#         incar_file.write(vaspstdincar.get_string())
#
#
# # Input files for vasp_ncl
#
#
# def _vasp_ncl_chempot(
#     structure: "pymatgen.core.Structure",
#     input_dir: Optional[str] = None,
#     incar_settings: Optional[dict] = None,
#     kpoints_settings: Optional[dict] = None,
#     potcar_settings: Optional[dict] = None,
# ) -> None:
#     """
#     Generates INCAR, POTCAR and KPOINTS for vasp_ncl chemical potentials
#     relaxation.
#
#     Take CONTCAR from vasp_std for POSCAR.:
#     Args:
#         structure (Structure object):
#             Structure to create input files for.
#         input_dir (str):
#             Folder in which to create vasp_ncl calculation inputs folder
#             (default: None)
#         incar_settings (dict):
#             Dictionary of user INCAR settings (AEXX, NCORE etc.) to override default settings.
#             Highly recommended to look at output INCARs or doped.vasp_input
#             source code, to see what the default INCAR settings are. Note that any flags that
#             aren't numbers or True/False need to be input as strings with quotation marks
#             (e.g. `{"ALGO": "All"}`).
#             (default: None)
#         kpoints_settings (dict):
#             Dictionary of user KPOINTS settings (in pymatgen Kpoints.from_dict() format). Common
#             options would be "generation_style": "Monkhorst" (rather than "Gamma"),
#             and/or "kpoints": [[3, 3, 1]] etc.
#             Default KPOINTS is Gamma-centred 2 x 2 x 2 mesh.
#             (default: None)
#         potcar_settings (dict):
#             Dictionary of user POTCAR settings to override default settings.
#             Highly recommended to look at `default_potcar_dict` from doped.vasp_input to see what
#             the (Pymatgen) syntax and doped default settings are.
#             (default: None).
#     """
#     # INCAR Parameters
#     vaspnclincardict = {
#         "# May need to change NELECT, NCORE, KPAR, AEXX, ENCUT, NUPDOWN": "variable parameters",
#         "NCORE": 12,
#         "KPAR": 2,
#         "AEXX": 0.25,
#         "ENCUT": 400,
#         "ICORELEVEL": "0 # Needed if using the Kumagai-Oba (eFNV) anisotropic charge correction",
#         "NSW": 0,
#         "LSORBIT": True,
#         "EDIFF": 1e-06,  # tight for final energy and converged DOS
#         "EDIFFG": -0.01,
#         "ALGO": "Normal # Change to All if ZHEGV, FEXCP/F or ZBRENT errors encountered",
#         "HFSCREEN": 0.2,
#         "IBRION": -1,
#         "ISYM": 0,
#         "ISMEAR": 0,
#         "LASPH": True,
#         "LHFCALC": True,
#         "LORBIT": 14,
#         "LREAL": False,
#         "LVHAR": "True # Needed if using the Freysoldt (FNV) charge correction scheme",
#         "LWAVE": True,
#         "NEDOS": 2000,
#         "NELM": 100,
#         "PREC": "Accurate",
#         "PRECFOCK": "Fast",
#         "SIGMA": 0.05,
#     }
#
#     # Directory
#     vaspnclinputdir = input_dir + "/vasp_ncl/" if input_dir else "VASP_Files/vasp_ncl/"
#     if not os.path.exists(vaspnclinputdir):
#         os.makedirs(vaspnclinputdir)
#
#     # POTCAR
#     potcar_dict = default_potcar_dict
#     if potcar_settings:
#         if "POTCAR_FUNCTIONAL" in potcar_settings:
#             potcar_dict["POTCAR_FUNCTIONAL"] = potcar_settings["POTCAR_FUNCTIONAL"]
#         if "POTCAR" in potcar_settings:
#             potcar_dict["POTCAR"].update(potcar_settings.pop("POTCAR"))
#     vaspnclinput = DictSet(structure, config_dict=potcar_dict)
#     vaspnclinput.potcar.write_file(vaspnclinputdir + "POTCAR")
#
#     if all(is_metal(element) for element in structure.composition.elements):
#         vaspnclincardict["ISMEAR"] = "2 # Metal, use Methfessel-Paxton smearing scheme"
#     if all(is_metal(element) for element in structure.composition.elements):
#         vaspnclincardict["SIGMA"] = 0.02
#
#     if incar_settings:
#         for k in incar_settings:  # check INCAR flags and warn if they don't exist (typos)
#             if k not in incar_params.keys():  # this code is taken from pymatgen.io.vasp.inputs
#                 warnings.warn(  # but only checking keys, not values so we can add comments etc
#                     "Cannot find %s from your user_incar_settings in the list of INCAR flags" % (k),
#                     BadIncarWarning,
#                 )
#         vaspnclincardict.update(incar_settings)
#
#     # KPOINTS
#     vaspnclkpointsdict = {
#         "comment": "Kpoints from doped.vasp_ncl_files",
#         "generation_style": "Gamma",  # Set to Monkhorst for Monkhorst-Pack generation
#         "kpoints": [[2, 2, 2]],
#     }
#     if kpoints_settings:
#         vaspnclkpointsdict.update(kpoints_settings)
#     vaspnclkpts = Kpoints.from_dict(vaspnclkpointsdict)
#     vaspnclkpts.write_file(vaspnclinputdir + "KPOINTS")
#
#     # INCAR
#     vaspnclincar = Incar.from_dict(vaspnclincardict)
#     with zopen(vaspnclinputdir + "INCAR", "wt") as incar_file:
#         incar_file.write(vaspnclincar.get_string())
