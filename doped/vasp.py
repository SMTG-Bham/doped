"""
Code to generate VASP defect calculation input files.
"""
import contextlib
import copy
import inspect
import os
import warnings
from functools import lru_cache
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
from monty.io import zopen
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
    get_defect_name_from_entry,
    name_defect_entries,
)
from doped.utils.parsing import (
    _get_bulk_supercell,
    _get_defect_supercell,
    _get_defect_supercell_bulk_site_coords,
)
from doped.utils.symmetry import _frac_coords_sort_func

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
    "EDIFFG": None,  # no ionic relaxation, remove to avoid confusion
    "IBRION": -1,  # no ionic relaxation
    "NSW": 0,  # no ionic relaxation
    "POTIM": None,  # no ionic relaxation, remove to avoid confusion
}


def _test_potcar_functional_choice(
    potcar_functional: UserPotcarFunctional = "PBE", symbols: Optional[List] = None
):
    """
    Check if the potcar functional choice needs to be changed to match those
    available.
    """
    test_potcar = None
    if symbols is None:
        symbols = ["Mg"]
    try:
        test_potcar = _get_potcar(tuple(symbols), potcar_functional=potcar_functional)
    except OSError as e:
        # try other functional choices:
        if potcar_functional.startswith("PBE"):
            for pbe_potcar_string in ["PBE", "PBE_52", "PBE_54"]:
                with contextlib.suppress(OSError):
                    potcar_functional = pbe_potcar_string
                    test_potcar = _get_potcar(tuple(symbols), potcar_functional=potcar_functional)
                    break

        if test_potcar is None:
            raise e

    return potcar_functional


@lru_cache(maxsize=1000)  # cache POTCAR generation to speed up generation and writing
def _get_potcar(potcar_symbols, potcar_functional) -> Potcar:
    return Potcar(list(potcar_symbols), functional=potcar_functional)


class DopedKpoints(Kpoints):
    """
    Custom implementation of ``Kpoints`` to handle encoding
    errors that can happen on some old HPCs/Linux systems.

    If an encoding error occurs upon file writing, then changes Γ to
    Gamma and Å to Angstrom in the ``KPOINTS`` comment.
    """

    def __repr__(self):
        """
        Returns a string representation of the Kpoints object, with encoding
        error handling.
        """
        try:
            with open("/dev/null", "w") as f:
                f.write(self.comment)  # breaks if encoding error will occur, so we rewrite
            return super().__repr__()

        except UnicodeEncodeError:
            self.comment = self.comment.replace("Å⁻³", "Angstrom^(-3)").replace("Γ", "Gamma")
            return super().__repr__()


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
        user_potcar_functional: UserPotcarFunctional = "PBE",
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
                or the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
                ``doped/VASP_sets`` folder, to see what the default INCAR settings are.
                Note that any flags that aren't numbers or True/False need to be input
                as strings with quotation marks (e.g. ``{"ALGO": "All"}``).
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
                ``doped/VASP_sets/PotcarSet.yaml`` for the default ``POTCAR`` set.
            poscar_comment (str):
                Comment line to use for POSCAR files. Default is defect name,
                fractional coordinates of initial site and charge state.
            **kwargs: Additional kwargs to pass to DictSet.
        """
        _ignore_pmg_warnings()
        self.charge_state = charge_state
        self.potcars = self._check_user_potcars(unperturbed_poscar=True, snb=False)
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

        # if "length" in user kpoint settings then pop reciprocal_density and use length instead
        if relax_set["KPOINTS"].get("length") or (
            user_kpoints_settings is not None
            and (
                "length" in user_kpoints_settings
                if isinstance(user_kpoints_settings, dict)
                else "length" in user_kpoints_settings.as_dict()
            )
        ):
            relax_set["KPOINTS"].pop("reciprocal_density", None)

        self.config_dict = self.CONFIG = relax_set  # avoid bug in pymatgen 2023.5.10, PR'd and fixed in
        # later versions

        # check POTCAR settings not in config dict format:
        if isinstance(user_potcar_settings, dict):
            if "POTCAR_FUNCTIONAL" in user_potcar_settings and user_potcar_functional == "PBE":
                # i.e. default
                user_potcar_functional = user_potcar_settings.pop("POTCAR_FUNCTIONAL")
            if "POTCAR" in user_potcar_settings:
                user_potcar_settings = user_potcar_settings["POTCAR"]

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
            incar_obj["NELECT"] = self.nelect
            if incar_obj["NELECT"] % 2 != 0:  # odd number of electrons
                incar_obj["NUPDOWN"] = 1
            else:
                # when writing VASP just resets this to 0 anyway:
                incar_obj["NUPDOWN"] = (
                    "0  # If defect has multiple spin-polarised states (e.g. bipolarons) could "
                    "also have triplet (NUPDOWN=2), but energy diff typically small."
                )

        except Exception as e:  # POTCARs unavailable, so NELECT and NUPDOWN can't be set
            # if it's a neutral defect, then this is ok (warn the user and write files), otherwise break
            if self.charge_state != 0:
                raise ValueError(
                    "NELECT (i.e. supercell charge) and NUPDOWN (i.e. spin state) INCAR flags cannot be "
                    "set due to the non-availability of POTCARs!\n(see the doped docs Installation page: "
                    "https://doped.readthedocs.io/en/latest/Installation.html for instructions on setting "
                    "this up)."
                ) from e

            warnings.warn(
                f"NUPDOWN (i.e. spin state) INCAR flag cannot be set due to the non-availability of "
                f"POTCARs!\n(see the doped docs Installation page: "
                f"https://doped.readthedocs.io/en/latest/Installation.html for instructions on setting "
                f"this up). As this is a neutral supercell, the INCAR file will be written without this "
                f"flag, but it is often important to explicitly set this spin state in VASP to avoid "
                f"unphysical solutions, and POTCARs are also needed to set the charge state (i.e. "
                f"NELECT) of charged defect supercells. Got error:\n{e!r}"
            )

        if "KPAR" in incar_obj and np.prod(self.kpoints.kpts[0]) == 1:
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
                self.user_potcar_functional, self.potcar_symbols
            )

        # use our own POTCAR generation function to expedite generation and writing
        return _get_potcar(tuple(self.potcar_symbols), self.user_potcar_functional)

    @property
    def poscar(self) -> Poscar:
        """
        Return Poscar object with comment.
        """
        unsorted_poscar = Poscar(self.structure, comment=self.poscar_comment)

        # check if POSCAR should be sorted:
        if len(unsorted_poscar.site_symbols) == len(set(unsorted_poscar.site_symbols)):
            # no duplicates, return poscar as is
            return unsorted_poscar

        return Poscar(self.structure, comment=self.poscar_comment, sort_structure=True)

    @property
    def kpoints(self):
        """
        Return kpoints object with comment.
        """
        pmg_kpoints = super().kpoints
        doped_kpoints = DopedKpoints.from_dict(pmg_kpoints.as_dict())
        kpt_density = self.config_dict.get("KPOINTS", {}).get("reciprocal_density", False)
        if (
            isinstance(self.user_kpoints_settings, dict)
            and "reciprocal_density" in self.user_kpoints_settings
        ):
            kpt_density = self.user_kpoints_settings.get("reciprocal_density", False)

        if kpt_density and all(i not in doped_kpoints.comment for i in ["doped", "ShakeNBreak"]):
            with contextlib.suppress(Exception):
                assert np.prod(doped_kpoints.kpts[0])  # check if it's a kpoint mesh (not custom kpoints)
                doped_kpoints.comment = f"KPOINTS from doped, with reciprocal_density = {kpt_density}/Å⁻³"

        elif all(i not in doped_kpoints.comment for i in ["doped", "ShakeNBreak"]):
            doped_kpoints.comment = "KPOINTS from doped"

        return doped_kpoints

    @property
    def nelect(self):
        """
        Number of electrons (``NELECT``) for the given structure and
        charge state.

        This is equal to the sum of valence electrons (ZVAL) of the
        ``POTCAR``s for each atom in the structure (supercell), minus
        the charge state.
        """
        neutral_nelect = super().nelect
        return neutral_nelect - self.charge_state

    def _check_user_potcars(self, unperturbed_poscar: bool = False, snb: bool = False) -> bool:
        """
        Check and warn the user if POTCARs are not set up with pymatgen.
        """
        potcars = any("VASP_PSP_DIR" in i for i in SETTINGS)
        if not potcars:
            potcar_warning_string = (
                "POTCAR directory not set up with pymatgen (see the doped docs Installation page: "
                "https://doped.readthedocs.io/en/latest/Installation.html for instructions on setting "
                "this up). This is required to generate `POTCAR` files and set the `NELECT` and "
                "`NUPDOWN` `INCAR` tags"
            )
            if unperturbed_poscar:
                if self.charge_state != 0:
                    warnings.warn(  # snb is hidden flag for ShakeNBreak (as the POSCARs aren't
                        # unperturbed in that case)
                        f"{potcar_warning_string}, so only {'' if snb else '(unperturbed) '}`POSCAR` and "
                        f"`KPOINTS` files will be generated."
                    )
                    return False

            elif self.charge_state != 0:  # only KPOINTS can be written so no good
                raise ValueError(f"{potcar_warning_string}, so no input files will be generated.")

            # if at this point, means charge_state == 0, so neutral INCAR can be generated
            warnings.warn(f"{potcar_warning_string}, so `POTCAR` files will not be generated.")

            return False

        return True

    def write_input(
        self,
        output_path: str,
        unperturbed_poscar: bool = True,
        make_dir_if_not_present: bool = True,
        include_cif: bool = False,
        potcar_spec: bool = False,
        zip_output: bool = False,
        snb: bool = False,
    ):
        """
        Writes out all input to a directory. Refactored slightly from ``pymatgen``
        ``DictSet.write_input()`` to allow checking of user ``POTCAR`` setup.

        Args:
            output_path (str): Directory to output the VASP input files.
            unperturbed_poscar (bool):
                If True, write the unperturbed defect POSCAR to the generated
                folder as well. (default: True)
            make_dir_if_not_present (bool): Set to True if you want the
                directory (and the whole path) to be created if it is not
                present. (default: True)
            include_cif (bool): Whether to write a CIF file in the output
                directory for easier opening by VESTA. (default: False)
            potcar_spec (bool): Instead of writing the POTCAR, write a "POTCAR.spec".
                This is intended to help sharing an input set with people who might
                not have a license to specific Potcar files. Given a "POTCAR.spec",
                the specific POTCAR file can be re-generated using pymatgen with the
                "generate_potcar" function in the pymatgen CLI. (default: False)
            zip_output (bool): Whether to zip each VASP input file written to the
                output directory. (default: False)
            snb (bool): If input structures are from ShakeNBreak (so POSCARs aren't
                'unperturbed'). (default: False)
        """
        if not potcar_spec:
            potcars = self._check_user_potcars(unperturbed_poscar=unperturbed_poscar, snb=snb)
        else:
            potcars = True

        if unperturbed_poscar and potcars:  # write everything, use DictSet.write_input()
            try:
                super().write_input(
                    output_path,
                    make_dir_if_not_present=make_dir_if_not_present,
                    include_cif=include_cif,
                    potcar_spec=potcar_spec,
                    zip_output=zip_output,
                )
            except ValueError as e:
                if str(e).startswith("NELECT") and potcar_spec:
                    with zopen(os.path.join(output_path, "POTCAR.spec"), "wt") as pot_spec_file:
                        pot_spec_file.write("\n".join(self.potcar_symbols))

                    self.kpoints.write_file(f"{output_path}/KPOINTS")
                    self.poscar.write_file(f"{output_path}/POSCAR")

        else:  # use `write_file()`s rather than `write_input()` to avoid writing POSCARs/POTCARs
            os.makedirs(output_path, exist_ok=True)

            # if not POTCARs and charge_state not 0, but unperturbed POSCAR is true, then skip INCAR
            # write attempt (unperturbed POSCARs and KPOINTS will be written, and user already warned):
            if potcars or self.charge_state == 0 or not unperturbed_poscar:
                self.incar.write_file(f"{output_path}/INCAR")

            if potcars:
                if potcar_spec:
                    with zopen(os.path.join(output_path, "POTCAR.spec"), "wt") as pot_spec_file:
                        pot_spec_file.write("\n".join(self.potcar_symbols))
                else:
                    self.potcar.write_file(f"{output_path}/POTCAR")

            self.kpoints.write_file(f"{output_path}/KPOINTS")

            if unperturbed_poscar:
                self.poscar.write_file(f"{output_path}/POSCAR")

    def __repr__(self):
        """
        Returns a string representation of the DefectDictet object.
        """
        attrs = {k for k in vars(self) if not k.startswith("_")}
        methods = {k for k in dir(self) if callable(getattr(self, k)) and not k.startswith("_")}
        properties = {
            name for name, value in inspect.getmembers(type(self)) if isinstance(value, property)
        }
        return (
            f"doped DefectDictSet with supercell composition {self.structure.composition}. "
            f"Available attributes:\n{attrs | properties}\n\nAvailable methods:\n{methods}"
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
    calculations from pymatgen ``DefectEntry`` (recommended) or ``Structure``
    objects.
    """

    def __init__(
        self,
        defect_entry: Union[DefectEntry, Structure],
        charge_state: Optional[int] = None,
        soc: Optional[bool] = None,
        user_incar_settings: Optional[dict] = None,
        user_kpoints_settings: Optional[Union[dict, Kpoints]] = None,
        user_potcar_functional: UserPotcarFunctional = "PBE",
        user_potcar_settings: Optional[dict] = None,
        **kwargs,
    ):
        r"""
        The supercell structure and charge state are taken from the ``DefectEntry``
        attributes, or if a ``Structure`` is provided, then from the
        ``defect_supercell`` and ``charge_state`` input parameters.

        Creates attributes:

        - ``DefectRelaxSet.vasp_gam``:
            ``DefectDictSet`` for Gamma-point only relaxation. Usually not needed if
            ShakeNBreak structure searching has been performed (recommended), unless
            only Γ-point `k`-point sampling is required (converged) for your system,
            and no vasp_std calculations with multiple `k`-points are required
            (determined from kpoints settings).
        - ``DefectRelaxSet.vasp_nkred_std``:
            ``DefectDictSet`` for relaxation with a kpoint mesh and using ``NKRED``. Not
            generated for GGA calculations (if ``LHFCALC`` is set to ``False`` in
            user_incar_settings) or if only Gamma `k`-point sampling is required.
        - ``DefectRelaxSet.vasp_std``:
            ``DefectDictSet`` for relaxation with a kpoint
            mesh, not using ``NKRED``. Not generated if only Gamma kpoint sampling is
            required.
        - ``DefectRelaxSet.vasp_ncl``:
            ``DefectDictSet`` for singlepoint (static) energy calculation with SOC
            included. Generated if ``soc=True``. If ``soc`` is not set, then by default
            is only generated for systems with a max atomic number (Z) >= 31 (i.e.
            further down the periodic table than Zn).

        where ``DefectDictSet`` is an extension of ``pymatgen``'s ``DictSet`` class for
        defect calculations, with ``incar``, ``poscar``, ``kpoints`` and ``potcar``
        attributes for the corresponding VASP defect calculations (see docstring).
        Also creates the corresponding ``bulk_vasp_...`` attributes for singlepoint
        (static) energy calculations of the bulk (pristine, defect-free)
        supercell. This needs to be calculated once with the same settings as the
        defect calculations, for the later calculation of defect formation energies.

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT`` settings,
        and ``PotcarSet.yaml`` for the default ``POTCAR`` settings.
        **These are reasonable defaults that _roughly_ match the typical values
        needed for accurate defect calculations, but usually will need to be
        modified for your specific system, such as converged ENCUT and KPOINTS,
        and NCORE / KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings should
        be consistent with those used for all defect and competing phase
        (chemical potential) calculations.

        Args:
            defect_entry (DefectEntry, Structure):
                doped/pymatgen DefectEntry or Structure (defect supercell) for
                which to generate ``DefectDictSet``\s for.
            charge_state (int):
                Charge state of the defect. Overrides ``DefectEntry.charge_state`` if
                ``DefectEntry`` is input.
            soc (bool):
                Whether to generate ``vasp_ncl`` DefectDictSet attribute for spin-orbit
                coupling singlepoint (static) energy calculations. If not set, then
                by default is set to True if the max atomic number (Z) in the
                structure is >= 31 (i.e. further down the periodic table than Zn),
                otherwise False.
            user_incar_settings (dict):
                Dictionary of user INCAR settings (AEXX, NCORE etc.) to override
                default settings. Highly recommended to look at output INCARs
                or the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
                ``doped/VASP_sets`` folder, to see what the default INCAR settings are.
                Note that any flags that aren't numbers or True/False need to be input
                as strings with quotation marks (e.g. ``{"ALGO": "All"}``).
                (default: None)
            user_kpoints_settings (dict or Kpoints):
                Dictionary of user KPOINTS settings (in pymatgen DictSet() format)
                e.g. {"reciprocal_density": 123}, or a Kpoints object, to use for the
                ``vasp_std``, ``vasp_nkred_std`` and ``vasp_ncl`` DefectDictSets (Γ-only for
                ``vasp_gam``). Default is Gamma-centred, reciprocal_density = 100 [Å⁻³].
            user_potcar_functional (str):
                POTCAR functional to use. Default is "PBE" and if this fails,
                tries "PBE_52", then "PBE_54".
            user_potcar_settings (dict):
                Override the default POTCARs, e.g. {"Li": "Li_sv"}. See
                ``doped/VASP_sets/PotcarSet.yaml`` for the default ``POTCAR`` set.
            **kwargs: Additional kwargs to pass to ``DefectDictSet()``.

        Attributes:
            vasp_gam (DefectDictSet):
                DefectDictSet for Gamma-point only relaxation. Usually not needed
                if ShakeNBreak structure searching has been performed
                (recommended), unless only Γ-point `k`-point sampling is required
                (converged) for your system, and no vasp_std calculations with
                multiple `k`-points are required (determined from kpoints settings).
            vasp_nkred_std (DefectDictSet):
                DefectDictSet for relaxation with a non-Γ-only kpoint mesh, using
                ``NKRED(X,Y,Z)`` INCAR tag(s) to downsample kpoints for the HF exchange
                part of the hybrid DFT calculation. Not generated for GGA calculations
                (if ``LHFCALC`` is set to ``False`` in user_incar_settings) or if only Gamma
                kpoint sampling is required.
            vasp_std (DefectDictSet):
                DefectDictSet for relaxation with a non-Γ-only kpoint mesh, not using
                ``NKRED``. Not generated if only Gamma kpoint sampling is required.
            vasp_ncl (DefectDictSet):
                DefectDictSet for singlepoint (static) energy calculation with SOC
                included. Generated if ``soc=True``. If ``soc`` is not set, then by default
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
                DefectDictSet for a `bulk` Γ-point-only singlepoint (static)
                supercell calculation. Often not used, as the bulk supercell only
                needs to be calculated once with the same settings as the final
                defect calculations, which may be with ``vasp_std`` or ``vasp_ncl``.
            bulk_vasp_nkred_std (DefectDictSet):
                DefectDictSet for a singlepoint (static) `bulk` ``vasp_std`` supercell
                calculation (i.e. with a non-Γ-only kpoint mesh) and ``NKRED(X,Y,Z)``
                INCAR tag(s) to downsample kpoints for the HF exchange part of the
                hybrid DFT calculation. Not generated for GGA calculations (if
                ``LHFCALC`` is set to ``False`` in user_incar_settings) or if only Gamma
                kpoint sampling is required.
            bulk_vasp_std (DefectDictSet):
                DefectDictSet for a singlepoint (static) `bulk` ``vasp_std`` supercell
                calculation with a non-Γ-only kpoint mesh, not using ``NKRED``. Not
                generated if only Gamma kpoint sampling is required.
            bulk_vasp_ncl (DefectDictSet):
                DefectDictSet for singlepoint (static) energy calculation of the `bulk`
                supercell with SOC included. Generated if ``soc=True``. If ``soc`` is not
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
            self.poscar_comment = self.dict_set_kwargs.pop("poscar_comment", None) or (
                f"{self.defect_supercell.formula} {'+' if self.charge_state > 0 else ''}"
                f"{self.charge_state}"
            )
            self.bulk_supercell = None

        elif isinstance(self.defect_entry, DefectEntry):
            self.defect_supercell = _get_defect_supercell(self.defect_entry)
            self.bulk_supercell = _get_bulk_supercell(self.defect_entry)
            if self.bulk_supercell is None:
                raise ValueError(
                    "Bulk supercell must be defined in DefectEntry object attributes. Both "
                    "DefectEntry.bulk_supercell and DefectEntry.bulk_entry are None!"
                )

            # get POSCAR comment:
            sc_frac_coords = _get_defect_supercell_bulk_site_coords(self.defect_entry)
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

            self.poscar_comment = self.dict_set_kwargs.pop("poscar_comment", None) or (
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
        Returns a DefectDictSet object for a VASP Γ-point-only (``vasp_gam``)
        defect supercell relaxation. Typically not needed if ShakeNBreak
        structure searching has been performed (recommended), unless only
        Γ-point `k`-point sampling is required (converged) for your system, and
        no vasp_std calculations with multiple `k`-points are required
        (determined from kpoints settings).

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT`` settings,
        and ``PotcarSet.yaml`` for the default ``POTCAR`` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings should
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
        using ``vasp_std`` (i.e. with a non-Γ-only kpoint mesh). Returns None and
        a warning if the input kpoint settings correspond to a Γ-only kpoint
        mesh (in which case ``vasp_gam`` should be used).

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT`` settings,
        and ``PotcarSet.yaml`` for the default ``POTCAR`` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings should
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
        using ``vasp_std`` (i.e. with a non-Γ-only kpoint mesh) and
        ``NKRED(X,Y,Z)`` INCAR tag(s) to downsample kpoints for the HF exchange
        part of hybrid DFT calculations, following the doped recommended defect
        calculation workflow (see docs). By default, sets ``NKRED(X,Y,Z)`` to 2
        or 3 in the directions for which the k-point grid is divisible by this
        factor. Returns None and a warning if the input kpoint settings
        correspond to a Γ-only kpoint mesh (in which case ``vasp_gam`` should be
        used) or for GGA calculations (if ``LHFCALC`` is set to ``False`` in
        user_incar_settings, in which case ``vasp_std`` should be used).

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT`` settings,
        and ``PotcarSet.yaml`` for the default ``POTCAR`` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.
        """
        if self.user_incar_settings.get("LHFCALC", True) is False:  # GGA
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
        using ``vasp_ncl``. If ``DefectRelaxSet.soc`` is False, then this returns
        None and a warning. If the ``soc`` parameter is not set when initializing
        ``DefectRelaxSet``, then this is set to True for systems with a max
        atomic number (Z) >= 31 (i.e. further down the periodic table than Zn),
        otherwise False.

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT`` settings,
        and ``PotcarSet.yaml`` for the default ``POTCAR`` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.
        """
        if not self.soc:
            return None

        user_incar_settings = copy.deepcopy(self.user_incar_settings)
        user_incar_settings.update(singleshot_incar_settings)
        user_incar_settings.update({"LSORBIT": True})  # ISYM already 0

        if "KPAR" not in user_incar_settings:
            user_incar_settings["KPAR"] = 2  # likely quicker with this, checked in DefectDictSet
            # if it's Γ-only vasp_ncl and reset to 1 accordingly

        # ignore warning with "KPAR", in case it's Γ-only
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "KPAR")  # `message` only needs to match start of message
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
        Returns a DefectDictSet object for a VASP `bulk` Γ-point-only
        (``vasp_gam``) singlepoint (static) supercell calculation. Often not
        used, as the bulk supercell only needs to be calculated once with the
        same settings as the final defect calculations, which is ``vasp_std`` if
        we have a non-Γ-only final k-point mesh, or ``vasp_ncl`` if SOC effects
        are being included. If the final converged k-point mesh is Γ-only, then
        ``bulk_vasp_gam`` should be used to calculate the singlepoint (static)
        bulk supercell reference energy. Can also sometimes be useful for the
        purpose of calculating defect formation energies at early stages of the
        typical ``vasp_gam`` -> ``vasp_nkred_std`` (if hybrid & non-Γ-only
        k-points) -> ``vasp_std`` (if non-Γ-only k-points) -> ``vasp_ncl`` (if SOC
        included) workflow, to obtain rough formation energy estimates and flag
        any potential issues with defect calculations early on.

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT`` settings,
        and ``PotcarSet.yaml`` for the default ``POTCAR`` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings should
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
        Returns a DefectDictSet object for a singlepoint (static) `bulk`
        ``vasp_std`` supercell calculation. Returns None and a warning if the
        input kpoint settings correspond to a Γ-only kpoint mesh (in which case
        ``(bulk_)vasp_gam`` should be used).

        The bulk supercell only needs to be calculated once with the same
        settings as the final defect calculations, which is ``vasp_std`` if we
        have a non-Γ-only final k-point mesh, ``vasp_ncl`` if SOC effects are
        being included (in which case ``bulk_vasp_ncl`` should be used for the
        singlepoint bulk supercell reference calculation), or ``vasp_gam`` if the
        final converged k-point mesh is Γ-only (in which case ``bulk_vasp_gam``
        should be used for the singlepoint bulk supercell reference
        calculation). Can also sometimes be useful for the purpose of
        calculating defect formation energies at midway stages of the typical
        ``vasp_gam`` -> ``vasp_nkred_std`` (if hybrid & non-Γ-only k-points) ->
        ``vasp_std`` (if non-Γ-only k-points) -> ``vasp_ncl`` (if SOC included)
        workflow, to obtain rough formation energy estimates and flag any
        potential issues with defect calculations early on.

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT`` settings,
        and ``PotcarSet.yaml`` for the default ``POTCAR`` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings should
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
        Returns a DefectDictSet object for a singlepoint (static) `bulk`
        ``vasp_std`` supercell calculation (i.e. with a non-Γ-only kpoint mesh)
        and ``NKRED(X,Y,Z)`` INCAR tag(s) to downsample kpoints for the HF
        exchange part of the hybrid DFT calculation. By default, sets
        ``NKRED(X,Y,Z)`` to 2 or 3 in the directions for which the k-point grid
        is divisible by this factor. Returns None and a warning if the input
        kpoint settings correspond to a Γ-only kpoint mesh (in which case
        ``(bulk_)vasp_gam`` should be used) or for GGA calculations (if ``LHFCALC``
        is set to ``False`` in user_incar_settings, in which case
        ``(bulk_)vasp_std`` should be used).

        The bulk supercell only needs to be calculated once with the same
        settings as the final defect calculations, which is ``vasp_std`` if we
        have a non-Γ-only final k-point mesh, ``vasp_ncl`` if SOC effects are
        being included (in which case ``bulk_vasp_ncl`` should be used for the
        singlepoint bulk supercell reference calculation), or ``vasp_gam`` if the
        final converged k-point mesh is Γ-only (in which case ``bulk_vasp_gam``
        should be used for the singlepoint bulk supercell reference
        calculation). Can also sometimes be useful for the purpose of
        calculating defect formation energies at midway stages of the typical
        ``vasp_gam`` -> ``vasp_nkred_std`` (if hybrid & non-Γ-only k-points) ->
        ``vasp_std`` (if non-Γ-only k-points) -> ``vasp_ncl`` (if SOC included)
        workflow, to obtain rough formation energy estimates and flag any
        potential issues with defect calculations early on.

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT`` settings,
        and ``PotcarSet.yaml`` for the default ``POTCAR`` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.
        """
        bulk_supercell: Structure = self._check_bulk_supercell_and_warn()
        if bulk_supercell is None:
            return None

        # check NKRED by running through ``vasp_nkred_std``:
        nkred_defect_dict_set = self.vasp_nkred_std

        if nkred_defect_dict_set is None:
            return None

        if nkred_defect_dict_set._check_user_potcars(unperturbed_poscar=True, snb=False):
            user_incar_settings = copy.deepcopy(self.user_incar_settings)
            if "KPAR" not in user_incar_settings:
                user_incar_settings["KPAR"] = 2  # multiple k-points, likely quicker with this
            user_incar_settings.update(singleshot_incar_settings)
            user_incar_settings.update(  # add NKRED settings
                {k: v for k, v in nkred_defect_dict_set.incar.as_dict().items() if "NKRED" in k}
            )
        else:
            user_incar_settings = {}

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
        Returns a DefectDictSet object for VASP `bulk` supercell singlepoint
        calculations with spin-orbit coupling (SOC) included (LSORBIT = True),
        using ``vasp_ncl``. If ``DefectRelaxSet.soc`` is False, then this returns
        None and a warning. If the ``soc`` parameter is not set when initializing
        ``DefectRelaxSet``, then this is set to True for systems with a max
        atomic number (Z) >= 31 (i.e. further down the periodic table than Zn),
        otherwise False.

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT`` settings,
        and ``PotcarSet.yaml`` for the default ``POTCAR`` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.
        """
        bulk_supercell: Structure = self._check_bulk_supercell_and_warn()
        if bulk_supercell is None:
            return None

        if not self.soc:
            return None

        user_incar_settings = copy.deepcopy(self.user_incar_settings)
        user_incar_settings.update(singleshot_incar_settings)
        user_incar_settings.update({"LSORBIT": True})  # ISYM already 0

        if "KPAR" not in user_incar_settings:
            user_incar_settings["KPAR"] = 2  # likely quicker with this, checked in DefectDictSet
            # if it's Γ-only vasp_ncl and reset to 1 accordingly

        # ignore warning with "KPAR", in case it's Γ-only
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "KPAR")  # `message` only needs to match start of message
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

    def _get_output_path(self, defect_dir: Optional[str] = None, subfolder: Optional[str] = None):
        if defect_dir is None:
            if self.defect_entry.name is None:
                self.defect_entry.name = get_defect_name_from_entry(self.defect_entry, relaxed=False)

            defect_dir = self.defect_entry.name

        return f"{defect_dir}/{subfolder}" if subfolder is not None else defect_dir

    def _write_vasp_xxx_files(
        self, defect_dir, subfolder, unperturbed_poscar, vasp_xxx_attribute, **kwargs
    ):
        output_path = self._get_output_path(defect_dir, subfolder)

        vasp_xxx_attribute.write_input(
            output_path,
            unperturbed_poscar,
            **kwargs,  # kwargs to allow POTCAR testing on GH Actions
        )

        if "bulk" not in defect_dir:  # not a bulk supercell
            self.defect_entry.to_json(f"{output_path}/{self.defect_entry.name}.json")

    def write_gam(
        self,
        defect_dir: Optional[str] = None,
        subfolder: Optional[str] = "vasp_gam",
        unperturbed_poscar: bool = True,
        bulk: bool = False,
        **kwargs,
    ):
        r"""
        Write the input files for VASP Γ-point-only (``vasp_gam``) defect
        supercell relaxation. Typically not recommended for use, as the
        recommended workflow is to perform ``vasp_gam`` calculations using
        ``ShakeNBreak`` for defect structure-searching and initial relaxations,
        but should be used if the final, converged `k`-point mesh is Γ-point-
        only. If ``bulk`` is True, the input files for a singlepoint calculation of
        the bulk supercell are also written to "{formula}_bulk/{subfolder}".

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT`` settings,
        and ``PotcarSet.yaml`` for the default ``POTCAR`` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.

        The ``DefectEntry`` object is also written to a ``json`` file in
        ``defect_dir`` to aid calculation provenance.

        Args:
            defect_dir (str):
                Folder in which to create the VASP defect calculation inputs.
                Default is to use the DefectEntry name (e.g. "Y_i_C4v_O1.92_+2"
                etc.), from ``self.defect_entry.name``. If this attribute is not
                set, it is automatically generated according to the doped
                convention (using ``get_defect_name_from_entry()``).
            subfolder (str):
                Output folder structure is ``<defect_dir>/<subfolder>`` where
                ``subfolder`` = 'vasp_gam' by default. Setting ``subfolder`` to
                ``None`` will write the ``vasp_gam`` input files directly to the
                ``<defect_dir>`` folder, with no subfolders created.
            unperturbed_poscar (bool):
                If True (default), write the unperturbed defect POSCAR to the
                generated folder as well. Typically not recommended for use, as
                the recommended workflow is to initially perform ``vasp_gam``
                ground-state structure searching using ShakeNBreak
                (https://shakenbreak.readthedocs.io), then continue the
                ``vasp(_nkred)_std`` relaxations from the ground-state structures
                (e.g. using ``-d vasp_nkred_std`` with `snb-groundstate` (CLI) or
                ``groundstate_folder="vasp_nkred_std"`` with
                ``write_groundstate_structure`` (Python API)).
                (default: True)
            bulk (bool):
                If True, the input files for a singlepoint calculation of the
                bulk supercell are also written to "{formula}_bulk/{subfolder}".
                (Default: False)
            **kwargs:
                Keyword arguments to pass to ``DefectDictSet.write_input()``.
        """
        if defect_dir is None:
            defect_dir = self.defect_entry.name

        self._write_vasp_xxx_files(
            defect_dir,
            subfolder,
            unperturbed_poscar=unperturbed_poscar,
            vasp_xxx_attribute=self.vasp_gam,
            **kwargs,
        )
        if bulk:
            bulk_supercell = self._check_bulk_supercell_and_warn()
            if bulk_supercell is None:
                return

            formula = bulk_supercell.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
            output_path = os.path.dirname(defect_dir) if "/" in defect_dir else "."
            self._write_vasp_xxx_files(
                f"{output_path}/{formula}_bulk",
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
        r"""
        Write the input files for a VASP defect supercell calculation using
        ``vasp_std`` (i.e. with a non-Γ-only kpoint mesh). By default, does not
        generate ``POSCAR`` (input structure) files, as these should be taken
        from the ``CONTCAR``\s of ``vasp_std`` relaxations using ``NKRED(X,Y,Z)``
        (originally from ``ShakeNBreak`` relaxations) if using hybrid DFT, or
        from ``ShakeNBreak`` calculations (via ``snb-groundstate -d vasp_std``)
        if using GGA, or, if not following the recommended structure-searching
        workflow, from the ``CONTCAR``\s of ``vasp_gam`` calculations. If
        unperturbed ``POSCAR`` files are desired, set ``unperturbed_poscar=True``.
        If bulk is True, the input files for a singlepoint calculation of the bulk
        supercell are also written to "{formula}_bulk/{subfolder}".

        Returns None and a warning if the input kpoint settings correspond to
        a Γ-only kpoint mesh (in which case ``vasp_gam`` should be used).

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT`` settings,
        and ``PotcarSet.yaml`` for the default ``POTCAR`` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.

        The ``DefectEntry`` object is also written to a ``json`` file in
        ``defect_dir`` to aid calculation provenance.

        Args:
            defect_dir (str):
                Folder in which to create the VASP defect calculation inputs.
                Default is to use the DefectEntry name (e.g. "Y_i_C4v_O1.92_+2"
                etc.), from ``self.defect_entry.name``. If this attribute is not
                set, it is automatically generated according to the doped
                convention (using ``get_defect_name_from_entry()``).
            subfolder (str):
                Output folder structure is ``<defect_dir>/<subfolder>`` where
                ``subfolder`` = 'vasp_std' by default. Setting ``subfolder`` to
                ``None`` will write the ``vasp_std`` input files directly to the
                ``<defect_dir>`` folder, with no subfolders created.
            unperturbed_poscar (bool):
                If True, write the unperturbed defect POSCAR to the generated
                folder as well. Not recommended, as the recommended workflow is
                to initially perform ``vasp_gam`` ground-state structure searching
                using ShakeNBreak (https://shakenbreak.readthedocs.io), then
                continue the ``vasp(_nkred)_std`` relaxations from the ground-state
                structures (e.g. using ``-d vasp_nkred_std`` with `snb-groundstate`
                (CLI) or ``groundstate_folder="vasp_nkred_std"`` with
                ``write_groundstate_structure`` (Python API)), first with NKRED if
                using hybrid DFT, then without NKRED.
                (default: False)
            bulk (bool):
                If True, the input files for a singlepoint calculation of the
                bulk supercell are also written to "{formula}_bulk/{subfolder}".
                (Default: False)
            **kwargs:
                Keyword arguments to pass to ``DefectDictSet.write_input()``.
        """
        if self.vasp_std is None:  # warns user if vasp_std is None
            return

        if defect_dir is None:
            defect_dir = self.defect_entry.name

        self._write_vasp_xxx_files(
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
            output_path = os.path.dirname(defect_dir) if "/" in defect_dir else "."
            self._write_vasp_xxx_files(
                f"{output_path}/{formula}_bulk",
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
        r"""
        Write the input files for defect calculations using ``vasp_std`` (i.e.
        with a non-Γ-only kpoint mesh) and ``NKRED(X,Y,Z)`` INCAR tag(s) to
        downsample kpoints for the HF exchange part of hybrid DFT calculations,
        following the doped recommended defect calculation workflow (see docs).
        By default, sets ``NKRED(X,Y,Z)`` to 2 or 3 in the directions for which
        the k-point grid is divisible by this factor.

        By default, does not generate ``POSCAR`` (input structure) files, as
        these should be taken from the ``CONTCAR``\s of ``ShakeNBreak`` calculations
        (via ``snb-groundstate -d vasp_nkred_std``) or, if not following the
        recommended structure-searching workflow, from the ``CONTCAR``\s of
        ``vasp_gam`` calculations. If unperturbed ``POSCAR`` files are desired, set
        ``unperturbed_poscar=True``.
        If bulk is True, the input files for a singlepoint calculation of the
        bulk supercell are also written to "{formula}_bulk/{subfolder}".

        Returns None and a warning if the input kpoint settings correspond to
        a Γ-only kpoint mesh (in which case ``vasp_gam`` should be used) or for
        GGA calculations (if ``LHFCALC`` is set to ``False`` in user_incar_settings,
        in which case ``vasp_std`` should be used).

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT`` settings,
        and ``PotcarSet.yaml`` for the default ``POTCAR`` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.

        The ``DefectEntry`` object is also written to a ``json`` file in
        ``defect_dir`` to aid calculation provenance.

        Args:
            defect_dir (str):
                Folder in which to create the VASP defect calculation inputs.
                Default is to use the DefectEntry name (e.g. "Y_i_C4v_O1.92_+2"
                etc.), from ``self.defect_entry.name``. If this attribute is not
                set, it is automatically generated according to the doped
                convention (using ``get_defect_name_from_entry()``).
            subfolder (str):
                Output folder structure is ``<defect_dir>/<subfolder>`` where
                ``subfolder`` = 'vasp_nkred_std' by default. Setting ``subfolder``
                to ``None`` will write the ``vasp_nkred_std`` input files directly
                to the ``<defect_dir>`` folder, with no subfolders created.
            unperturbed_poscar (bool):
                If True, write the unperturbed defect POSCAR to the generated
                folder as well. Not recommended, as the recommended workflow is
                to initially perform ``vasp_gam`` ground-state structure searching
                using ShakeNBreak (https://shakenbreak.readthedocs.io), then continue
                the ``vasp(_nkred)_std`` relaxations from the ground-state structures
                (e.g. using ``-d vasp_nkred_std`` with `snb-groundstate` (CLI) or
                ``groundstate_folder="vasp_nkred_std"`` with
                ``write_groundstate_structure`` (Python API)).
                (default: False)
            bulk (bool):
                If True, the input files for a singlepoint calculation of the
                bulk supercell are also written to "{formula}_bulk/{subfolder}".
                (Default: False)
            **kwargs:
                Keyword arguments to pass to ``DefectDictSet.write_input()``.
        """
        if self.vasp_nkred_std is None:  # warn user if vasp_nkred_std is None
            warnings.warn(
                "`LHFCALC` is set to `False` in user_incar_settings, so `vasp_nkred_std` is None (as "
                "NKRED acts to downsample the Fock exchange potential in _hybrid DFT_ calculations), "
                "and so `vasp_std` should be used instead."
            )
            return

        if defect_dir is None:
            defect_dir = self.defect_entry.name

        self._write_vasp_xxx_files(
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
            output_path = os.path.dirname(defect_dir) if "/" in defect_dir else "."
            self._write_vasp_xxx_files(
                f"{output_path}/{formula}_bulk",
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
        r"""
        Write the input files for VASP defect supercell singlepoint
        calculations with spin-orbit coupling (SOC) included (LSORBIT = True),
        using ``vasp_ncl``. By default, does not generate ``POSCAR`` (input
        structure) files, as these should be taken from the ``CONTCAR``\s of
        ``vasp_std`` relaxations (originally from ``ShakeNBreak`` structure-
        searching relaxations), or directly from ``ShakeNBreak`` calculations
        (via ``snb-groundstate -d vasp_ncl``) if only Γ-point reciprocal space
        sampling is required. If unperturbed ``POSCAR`` files are desired, set
        ``unperturbed_poscar=True``.

        If ``DefectRelaxSet.soc`` is False, then this returns None and a warning.
        If the ``soc`` parameter is not set when initializing ``DefectRelaxSet``,
        then it is set to True for systems with a max atomic number (Z) >= 31
        (i.e. further down the periodic table than Zn), otherwise False.
        If bulk is True, the input files for a singlepoint calculation of the
        bulk supercell are also written to "{formula}_bulk/{subfolder}".

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT`` settings,
        and ``PotcarSet.yaml`` for the default ``POTCAR`` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.

        The ``DefectEntry`` object is also written to a ``json`` file in
        ``defect_dir`` to aid calculation provenance.

        Args:
            defect_dir (str):
                Folder in which to create the VASP defect calculation inputs.
                Default is to use the DefectEntry name (e.g. "Y_i_C4v_O1.92_+2"
                etc.), from ``self.defect_entry.name``. If this attribute is not
                set, it is automatically generated according to the doped
                convention (using ``get_defect_name_from_entry()``).
            subfolder (str):
                Output folder structure is ``<defect_dir>/<subfolder>`` where
                ``subfolder`` = 'vasp_ncl' by default. Setting ``subfolder`` to
                ``None`` will write the ``vasp_ncl`` input files directly to the
                ``<defect_dir>`` folder, with no subfolders created.
            unperturbed_poscar (bool):
                If True, write the unperturbed defect POSCAR to the generated
                folder as well. Not recommended, as the recommended workflow is
                to initially perform ``vasp_gam`` ground-state structure searching
                using ShakeNBreak (https://shakenbreak.readthedocs.io), then continue
                the ``vasp(_nkred)_std`` relaxations from the ground-state structures
                (e.g. using ``-d vasp_nkred_std`` with `snb-groundstate` (CLI) or
                ``groundstate_folder="vasp_nkred_std"`` with
                ``write_groundstate_structure`` (Python API)), first with NKRED if
                using hybrid DFT, then without, then use the ``vasp_std`` ``CONTCAR``\s
                as the input structures for the final ``vasp_ncl`` singlepoint calculations.
                (default: False)
            bulk (bool):
                If True, the input files for a singlepoint calculation of the
                bulk supercell are also written to "{formula}_bulk/{subfolder}".
                (Default: False)
            **kwargs:
                Keyword arguments to pass to ``DefectDictSet.write_input()``.
        """
        if self.vasp_ncl is None:
            warnings.warn(
                "DefectRelaxSet.soc is False, so `vasp_ncl` is None (i.e. no `vasp_ncl` input files "
                "have been generated). If SOC calculations are desired, set soc=True when initializing "
                "DefectRelaxSet. Otherwise, use bulk_vasp_std or bulk_vasp_gam instead."
            )
            return

        if defect_dir is None:
            defect_dir = self.defect_entry.name

        self._write_vasp_xxx_files(
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
            output_path = os.path.dirname(defect_dir) if "/" in defect_dir else "."
            self._write_vasp_xxx_files(
                f"{output_path}/{formula}_bulk",
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
        r"""
        Write all VASP input files to subfolders in the ``defect_dir`` folder.

        The following subfolders are generated:

        - ``vasp_nkred_std``:
            Defect relaxation with a kpoint mesh and using ``NKRED``. Not
            generated for GGA calculations (if ``LHFCALC`` is set to ``False``
            in ``user_incar_settings``) or if only Γ-point sampling required.
        - ``vasp_std``:
            Defect relaxation with a kpoint mesh, not using ``NKRED``. Not
            generated if only Γ-point sampling required.
        - ``vasp_ncl``:
            Singlepoint (static) energy calculation with SOC included.
            Generated if ``soc=True``. If ``soc`` is not set, then by default is
            only generated for systems with a max atomic number (Z) >= 31
            (i.e. further down the periodic table than Zn).

        If vasp_gam=True (not recommended) or self.vasp_std = None (i.e. Γ-only
        `k`-point sampling converged for the kpoints settings used), then also
        outputs:

        - ``vasp_gam``:
            Γ-point only defect relaxation. Usually not needed if ShakeNBreak
            structure searching has been performed (recommended).

        By default, does not generate a ``vasp_gam`` folder unless ``self.vasp_std``
        is None (i.e. only Γ-point sampling required for this system), as
        ``vasp_gam`` calculations should be performed using ``ShakeNBreak`` for
        defect structure-searching and initial relaxations. If ``vasp_gam`` files
        are desired, set ``vasp_gam=True``.

        By default, ``POSCAR`` files are not generated for the ``vasp_(nkred_)std``
        (and ``vasp_ncl`` if ``self.soc`` is True) folders, as these should
        be taken from ``ShakeNBreak`` calculations (via
        ``snb-groundstate -d vasp_nkred_std``) or, if not following the recommended
        structure-searching workflow, from the ``CONTCAR``\s of ``vasp_gam``
        calculations. If including SOC effects (``self.soc = True``), then the
        ``vasp_std`` ``CONTCAR``\s should be used as the ``vasp_ncl`` ``POSCAR``\s.
        If unperturbed ``POSCAR`` files are desired for the ``vasp_(nkred_)std``
        (and ``vasp_ncl``) folders, set ``unperturbed_poscar=True``.

        Input files for the singlepoint (static) bulk supercell reference
        calculation are also written to "{formula}_bulk/{subfolder}" if ``bulk``
        is True (False by default), where ``subfolder`` corresponds to the final
        (highest accuracy) VASP calculation in the workflow (i.e. ``vasp_ncl`` if
        ``self.soc=True``, otherwise ``vasp_std`` or ``vasp_gam`` if only Γ-point
        reciprocal space sampling is required). If ``bulk = "all"``, then the
        input files for all VASP calculations in the workflow are written to
        the bulk supercell folder.

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT`` settings,
        and ``PotcarSet.yaml`` for the default ``POTCAR`` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.

        The ``DefectEntry`` object is also written to a ``json`` file in
        ``defect_dir`` to aid calculation provenance.

        Args:
            defect_dir (str):
                Folder in which to create the VASP defect calculation inputs.
                Default is to use the DefectEntry name (e.g. "Y_i_C4v_O1.92_+2"
                etc.), from ``self.defect_entry.name``. If this attribute is not
                set, it is automatically generated according to the doped
                convention (using ``get_defect_name_from_entry()``).
                Output folder structure is ``<defect_dir>/<subfolder>`` where
                ``subfolder`` is the name of the corresponding VASP program to run
                (e.g. ``vasp_std``).
            unperturbed_poscar (bool):
                If True, write the unperturbed defect POSCARs to the generated
                folders as well. Not recommended, as the recommended workflow is
                to initially perform ``vasp_gam`` ground-state structure searching
                using ShakeNBreak (https://shakenbreak.readthedocs.io), then continue
                the ``vasp(_nkred)_std`` relaxations from the ground-state structures
                (e.g. using ``-d vasp_nkred_std`` with `snb-groundstate` (CLI) or
                ``groundstate_folder="vasp_nkred_std"`` with
                ``write_groundstate_structure`` (Python API)) first with NKRED if
                using hybrid DFT, then without, then use the ``vasp_std`` ``CONTCAR``\s
                as the input structures for the final ``vasp_ncl`` singlepoint
                calculations.
                (default: False)
            vasp_gam (bool):
                If True, write the ``vasp_gam`` input files, with unperturbed defect
                POSCAR. Not recommended, as the recommended workflow is to initially
                perform ``vasp_gam`` ground-state structure searching using ShakeNBreak
                (https://shakenbreak.readthedocs.io), then continue the ``vasp_std``
                relaxations from the SnB ground-state structures.
                (default: False)
            bulk (bool, str):
                If True, the input files for a singlepoint calculation of the
                bulk supercell are also written to "{formula}_bulk/{subfolder}",
                where ``subfolder`` corresponds to the final (highest accuracy)
                VASP calculation in the workflow (i.e. ``vasp_ncl`` if ``self.soc=True``,
                otherwise ``vasp_std`` or ``vasp_gam`` if only Γ-point reciprocal space
                sampling is required). If ``bulk = "all"`` then the input files for
                all VASP calculations in the workflow (``vasp_gam``, ``vasp_nkred_std``,
                ``vasp_std``, ``vasp_ncl`` (if applicable)) are written to the bulk
                supercell folder.
                (Default: False)
            **kwargs:
                Keyword arguments to pass to ``DefectDictSet.write_input()``.
        """
        # check `bulk` input:
        bulk_vasp = []
        if bulk and isinstance(bulk, str) and bulk.lower() == "all":
            bulk_vasp = ["vasp_gam", "vasp_nkred_std", "vasp_std", "vasp_ncl"]
        elif bulk and not isinstance(bulk, str) and bulk is not True:
            raise ValueError("Unrecognised input for `bulk` argument. Must be True, False, or 'all'.")

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

            if vasp_gam or self.vasp_std is None:  # if vasp_std is None, write vasp_gam
                self.write_gam(
                    defect_dir=defect_dir,
                    bulk=any("vasp_gam" in vasp_type for vasp_type in bulk_vasp),
                    unperturbed_poscar=unperturbed_poscar or vasp_gam,  # unperturbed poscar only if
                    # `vasp_gam` explicitly set to True
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

    def __repr__(self):
        """
        Returns a string representation of the DefectRelaxSet object.
        """
        formula = self.bulk_supercell.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
        attrs = {k for k in vars(self) if not k.startswith("_")}
        methods = {k for k in dir(self) if callable(getattr(self, k)) and not k.startswith("_")}
        properties = {
            name for name, value in inspect.getmembers(type(self)) if isinstance(value, property)
        }
        return (
            f"doped DefectRelaxSet for bulk composition {formula}, and defect entry "
            f"{self.defect_entry.name}. Available attributes:\n{attrs | properties}\n\n"
            f"Available methods:\n{methods}"
        )


class DefectsSet(MSONable):
    """
    An object for generating input files for VASP defect calculations from
    doped/pymatgen ``DefectEntry`` objects.
    """

    def __init__(
        self,
        defect_entries: Union[DefectsGenerator, Dict[str, DefectEntry], List[DefectEntry], DefectEntry],
        soc: Optional[bool] = None,
        user_incar_settings: Optional[dict] = None,
        user_kpoints_settings: Optional[Union[dict, Kpoints]] = None,
        user_potcar_functional: UserPotcarFunctional = "PBE",
        user_potcar_settings: Optional[dict] = None,
        **kwargs,  # to allow POTCAR testing on GH Actions
    ):
        r"""
        Creates a dictionary of: {defect_species: DefectRelaxSet}.

        DefectRelaxSet has the attributes:

        - ``DefectRelaxSet.vasp_gam``:
            ``DefectDictSet`` for Gamma-point only relaxation. Usually not needed if
            ShakeNBreak structure searching has been performed (recommended), unless
            only Γ-point `k`-point sampling is required (converged) for your system,
            and no vasp_std calculations with multiple `k`-points are required
            (determined from kpoints settings).
        - ``DefectRelaxSet.vasp_nkred_std``:
            ``DefectDictSet`` for relaxation with a kpoint mesh and using ``NKRED``.
            Not generated for GGA calculations (if ``LHFCALC`` is set to ``False`` in
            ``user_incar_settings``) or if only Gamma kpoint sampling is required.
        - ``DefectRelaxSet.vasp_std``:
            ``DefectDictSet`` for relaxation with a kpoint mesh, not using ``NKRED``.
            Not generated if only Gamma kpoint sampling is required.
        - ``DefectRelaxSet.vasp_ncl``:
            ``DefectDictSet`` for singlepoint (static) energy calculation with SOC
            included. Generated if ``soc=True``. If ``soc`` is not set, then by default
            is only generated for systems with a max atomic number (Z) >= 31
            (i.e. further down the periodic table than Zn).

        where ``DefectDictSet`` is an extension of ``pymatgen``'s ``DictSet`` class for
        defect calculations, with ``incar``, ``poscar``, ``kpoints`` and ``potcar``
        attributes for the corresponding VASP defect calculations (see docstring).
        Also creates the corresponding ``bulk_vasp_...`` attributes for singlepoint
        (static) energy calculations of the bulk (pristine, defect-free) supercell.
        This needs to be calculated once with the same settings as the final defect
        calculations, for the later calculation of defect formation energies.

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` settings, and
        ``PotcarSet.yaml`` for the default ``POTCAR`` settings.

        Note that any changes to the default ``INCAR``/``POTCAR`` settings should
        be consistent with those used for all defect and competing phase (chemical
        potential) calculations.

        Args:
            defect_entries (``DefectsGenerator``, dict/list of ``DefectEntry``\s, or ``DefectEntry``):
                Either a ``DefectsGenerator`` object, or a dictionary/list of
                ``DefectEntry``\s, or a single ``DefectEntry`` object, for which
                to generate VASP input files.
                If a ``DefectsGenerator`` object or a dictionary (->
                {defect_species: DefectEntry}), the defect folder names will be
                set equal to ``defect_species``. If a list or single ``DefectEntry``
                object is provided, the defect folder names will be set equal to
                ``DefectEntry.name`` if the ``name`` attribute is set, otherwise
                generated according to the ``doped`` convention (see doped.generation).
                Defect charge states are taken from ``DefectEntry.charge_state``.
            soc (bool):
                Whether to generate ``vasp_ncl`` DefectDictSet attribute for spin-orbit
                coupling singlepoint (static) energy calculations. If not set, then
                by default is set to True if the max atomic number (Z) in the
                structure is >= 31 (i.e. further down the periodic table than Zn).
            user_incar_settings (dict):
                Dictionary of user INCAR settings (AEXX, NCORE etc.) to override
                default settings. Highly recommended to look at output INCARs or the
                ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the ``doped/VASP_sets``
                folder, to see what the default INCAR settings are. Note that any
                flags that aren't numbers or True/False need to be input as strings
                with quotation marks (e.g. ``{"ALGO": "All"}``).
                (default: None)
            user_kpoints_settings (dict or Kpoints):
                Dictionary of user KPOINTS settings (in pymatgen DictSet() format)
                e.g. {"reciprocal_density": 123}, or a Kpoints object, to use for the
                ``vasp_std``, ``vasp_nkred_std`` and ``vasp_ncl`` DefectDictSets (Γ-only for
                ``vasp_gam``). Default is Gamma-centred, reciprocal_density = 100 [Å⁻³].
            user_potcar_functional (str):
                POTCAR functional to use. Default is "PBE" and if this fails, tries
                "PBE_52", then "PBE_54".
            user_potcar_settings (dict):
                Override the default POTCARs, e.g. {"Li": "Li_sv"}. See
                ``doped/VASP_setsPotcarSet.yaml`` for the default ``POTCAR`` set.
            **kwargs: Additional kwargs to pass to each ``DefectRelaxSet()``.

        Attributes:
            defect_sets (Dict):
                Dictionary of {defect_species: ``DefectRelaxSet``}.
            defect_entries (Dict):
                Dictionary of {defect_species: DefectEntry} for the input defect
                species, for which to generate VASP input files.
            bulk_vasp_gam (DefectDictSet):
                DefectDictSet for a `bulk` Γ-point-only singlepoint (static)
                supercell calculation. Often not used, as the bulk supercell only
                needs to be calculated once with the same settings as the final
                defect calculations, which may be with ``vasp_std`` or ``vasp_ncl``.
            bulk_vasp_nkred_std (DefectDictSet):
                DefectDictSet for a singlepoint (static) `bulk` ``vasp_std`` supercell
                calculation (i.e. with a non-Γ-only kpoint mesh) and ``NKRED(X,Y,Z)``
                INCAR tag(s) to downsample kpoints for the HF exchange part of the
                hybrid DFT calculation. Not generated for GGA calculations (if
                ``LHFCALC`` is set to ``False`` in user_incar_settings) or if only Gamma
                kpoint sampling is required.
            bulk_vasp_std (DefectDictSet):
                DefectDictSet for a singlepoint (static) `bulk` ``vasp_std`` supercell
                calculation with a non-Γ-only kpoint mesh, not using ``NKRED``. Not
                generated if only Gamma kpoint sampling is required.
            bulk_vasp_ncl (DefectDictSet):
                DefectDictSet for singlepoint (static) energy calculation of the `bulk`
                supercell with SOC included. Generated if ``soc=True``. If ``soc`` is not
                set, then by default is only generated for systems with a max atomic
                number (Z) >= 31 (i.e. further down the periodic table than Zn).
            json_obj (Union[Dict, DefectsGenerator]):
                Either the DefectsGenerator object if input ``defect_entries`` is a
                ``DefectsGenerator`` object, otherwise the ``defect_entries`` dictionary,
                which will be written to file when ``write_files()`` is called, to
                aid calculation provenance.
            json_name (str):
                Name of the JSON file to save the ``json_obj`` to.

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
                defect_supercell = _get_defect_supercell(defect_entry)
                if defect_supercell is not None:
                    return defect_supercell.atomic_numbers

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

        # set bulk vasp attributes:
        if not self.defect_sets:
            raise ValueError(
                "No `DefectRelaxSet` objects created, indicating problems with the `DefectsSet` "
                "input/creation!"
            )
        defect_relax_set = list(self.defect_sets.values())[-1]
        self.bulk_vasp_gam = defect_relax_set.bulk_vasp_gam
        self.bulk_vasp_nkred_std = defect_relax_set.bulk_vasp_nkred_std
        self.bulk_vasp_std = defect_relax_set.bulk_vasp_std
        self.bulk_vasp_ncl = defect_relax_set.bulk_vasp_ncl

    def _format_defect_entries_input(
        self,
        defect_entries: Union[DefectsGenerator, Dict[str, DefectEntry], List[DefectEntry], DefectEntry],
    ) -> Tuple[Dict[str, DefectEntry], str, Union[Dict[str, DefectEntry], DefectsGenerator]]:
        r"""
        Helper function to format input ``defect_entries`` into a named
        dictionary of ``DefectEntry`` objects. Also returns the name of the JSON
        file and object to serialise when writing the VASP input to files. This
        is the DefectsGenerator object if ``defect_entries`` is a
        ``DefectsGenerator`` object, otherwise the dictionary of ``DefectEntry``
        objects.

        Args:
            defect_entries (``DefectsGenerator``, dict/list of ``DefectEntry``\s, or ``DefectEntry``):
                Either a ``DefectsGenerator`` object, or a dictionary/list of
                ``DefectEntry``\s, or a single ``DefectEntry`` object, for which
                to generate VASP input files.
                If a ``DefectsGenerator`` object or a dictionary (->
                {defect_species: DefectEntry}), the defect folder names will be
                set equal to ``defect_species``. If a list or single ``DefectEntry``
                object is provided, the defect folder names will be set equal to
                ``DefectEntry.name`` if the ``name`` attribute is set, otherwise
                generated according to the ``doped`` convention (see doped.generation).
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
            with contextlib.suppress(AttributeError, TypeError):  # sort by conventional cell
                # fractional coordinates if these are defined, to aid deterministic naming
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
        defect_species, defect_relax_set, output_path, unperturbed_poscar, vasp_gam, bulk, kwargs = args
        defect_dir = os.path.join(output_path, defect_species)
        defect_relax_set.write_all(
            defect_dir=defect_dir,
            unperturbed_poscar=unperturbed_poscar,
            vasp_gam=vasp_gam,
            bulk=bulk,
            **kwargs,
        )

    def write_files(
        self,
        output_path: str = ".",
        unperturbed_poscar: bool = False,
        vasp_gam: bool = False,
        bulk: Union[bool, str] = True,
        processes: Optional[int] = None,
        **kwargs,
    ):
        r"""
        Write VASP input files to folders for all defects in
        ``self.defect_entries``. Folder names are set to the key of the
        DefectRelaxSet in ``self.defect_sets`` (same as self.defect_entries keys,
        see ``DefectsSet`` docstring).

        For each defect folder, the following subfolders are generated:

        - ``vasp_nkred_std``:
            Defect relaxation with a kpoint mesh and using ``NKRED``. Not generated
            for GGA calculations (if ``LHFCALC`` is set to ``False`` in
            ``user_incar_settings``) or if only Γ-point sampling required.
        - ``vasp_std``:
            Defect relaxation with a kpoint mesh, not using ``NKRED``. Not
            generated if only Γ-point sampling required.
        - ``vasp_ncl``:
            Singlepoint (static) energy calculation with SOC included. Generated if
            ``soc=True``. If ``soc`` is not set, then by default is only generated
            for systems with a max atomic number (Z) >= 31 (i.e. further down the
            periodic table than Zn).

        If ``vasp_gam=True`` (not recommended) or ``self.vasp_std = None`` (i.e. Γ-only
        `k`-point sampling converged for the kpoints settings used), then also
        outputs:

        - ``vasp_gam``:
            Γ-point only defect relaxation. Usually not needed if ShakeNBreak structure
            searching has been performed (recommended).

        By default, does not generate a ``vasp_gam`` folder unless
        ``DefectRelaxSet.vasp_std`` is None (i.e. only Γ-point sampling required
        for this system), as ``vasp_gam`` calculations should be performed using
        ``ShakeNBreak`` for defect structure-searching and initial relaxations.
        If ``vasp_gam`` files are desired, set ``vasp_gam=True``.

        By default, ``POSCAR`` files are not generated for the ``vasp_(nkred_)std``
        (and ``vasp_ncl`` if ``self.soc`` is True) folders, as these should
        be taken from ``vasp_gam`` ``ShakeNBreak`` calculations (via
        ``snb-groundstate -d vasp_nkred_std``), some other structure-searching
        approach or, if not following the recommended structure-searching workflow,
        from the ``CONTCAR``\s of ``vasp_gam`` calculations. If including SOC
        effects (``self.soc = True``), then the ``vasp_std`` ``CONTCAR``\s
        should be used as the ``vasp_ncl`` ``POSCAR``\s. If unperturbed
        ``POSCAR`` files are desired for the ``vasp_(nkred_)std`` (and ``vasp_ncl``)
        folders, set ``unperturbed_poscar=True``.

        Input files for the singlepoint (static) bulk supercell reference
        calculation are also written to "{formula}_bulk/{subfolder}" if ``bulk``
        is ``True`` (default), where ``subfolder`` corresponds to the final (highest
        accuracy) VASP calculation in the workflow (i.e. ``vasp_ncl`` if
        ``self.soc=True``, otherwise ``vasp_std`` or ``vasp_gam`` if only Γ-point
        reciprocal space sampling is required). If ``bulk = "all"``, then the
        input files for all VASP calculations (gam/std/ncl) are written to the bulk
        supercell folder, or if ``bulk = False``, then no bulk folder is created.

        The ``DefectEntry`` objects are also written to ``json`` files in the defect
        folders, as well as ``self.defect_entries`` (``self.json_obj``) in the top
        folder, to aid calculation provenance.

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT`` settings,
        and ``PotcarSet.yaml`` for the default ``POTCAR`` settings. **These are
        reasonable defaults that _roughly_ match the typical values needed for
        accurate defect calculations, but usually will need to be modified for
        your specific system, such as converged ENCUT and KPOINTS, and NCORE /
        KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings should
        be consistent with those used for all defect and competing phase (
        chemical potential) calculations.

        Args:
            output_path (str):
                Folder in which to create the VASP defect calculation folders.
                Default is the current directory ("."). Output folder structure
                is ``<output_path>/<defect_species>/<subfolder>`` where
                ``defect_species`` is the key of the DefectRelaxSet in
                ``self.defect_sets`` (same as ``self.defect_entries`` keys, see
                ``DefectsSet`` docstring) and ``subfolder`` is the name of the
                corresponding VASP program to run (e.g. ``vasp_std``).
            unperturbed_poscar (bool):
                If True, write the unperturbed defect POSCARs to the generated
                folders as well. Not recommended, as the recommended workflow is
                to initially perform ``vasp_gam`` ground-state structure searching
                using ShakeNBreak (https://shakenbreak.readthedocs.io), then continue
                the ``vasp(_nkred)_std`` relaxations from the ground-state structures
                (e.g. using ``-d vasp_nkred_std`` with `snb-groundstate` (CLI) or
                ``groundstate_folder="vasp_nkred_std"`` with
                ``write_groundstate_structure`` (Python API)) first with NKRED if
                using hybrid DFT, then without, then use the ``vasp_std`` ``CONTCAR``\s
                as the input structures for the final ``vasp_ncl`` singlepoint
                calculations.
                (default: False)
            vasp_gam (bool):
                If True, write the ``vasp_gam`` input files, with unperturbed defect
                POSCARs. Not recommended, as the recommended workflow is to initially
                perform ``vasp_gam`` ground-state structure searching using ShakeNBreak
                (https://shakenbreak.readthedocs.io), then continue the ``vasp_std``
                relaxations from the SnB ground-state structures.
                (default: False)
            bulk (bool, str):
                If True, the input files for a singlepoint calculation of the
                bulk supercell are also written to "{formula}_bulk/{subfolder}",
                where ``subfolder`` corresponds to the final (highest accuracy)
                VASP calculation in the workflow (i.e. ``vasp_ncl`` if ``self.soc=True``,
                otherwise ``vasp_std`` or ``vasp_gam`` if only Γ-point reciprocal space
                sampling is required). If ``bulk = "all"`` then the input files for
                all VASP calculations in the workflow (``vasp_gam``, ``vasp_nkred_std``,
                ``vasp_std``, ``vasp_ncl`` (if applicable)) are written to the bulk
                supercell folder.
                (Default: False)
            processes (int):
                Number of processes to use for multiprocessing for file writing.
                If not specified (default), then is dynamically set to the optimal
                value for the number of folders to write. (Default: None)
            **kwargs:
                Keyword arguments to pass to ``DefectDictSet.write_input()``.
        """
        # TODO: If POTCARs not setup, warn and only write neutral defect folders, with INCAR, KPOINTS and
        #  (if unperturbed_poscar) POSCAR? And bulk

        args_list = [
            (
                defect_species,
                defect_relax_set,
                output_path,
                unperturbed_poscar,
                vasp_gam,
                bulk if i == len(self.defect_sets) - 1 else False,  # write bulk folder(s) for last defect
                kwargs,
            )
            for i, (defect_species, defect_relax_set) in enumerate(self.defect_sets.items())
        ]
        if processes is None:  # best setting for number of processes, from testing
            processes = min(round(len(args_list) / 30), cpu_count() - 1)

        if processes > 1:
            with Pool(processes=processes or cpu_count() - 1) as pool:
                for _ in tqdm(
                    pool.imap(self._write_defect, args_list),
                    total=len(args_list),
                    desc="Generating and writing input files",
                ):
                    pass
        else:
            for args in tqdm(args_list, desc="Generating and writing input files"):
                self._write_defect(args)

        dumpfn(self.json_obj, os.path.join(output_path, self.json_name))

    def __repr__(self):
        """
        Returns a string representation of the DefectsSet object.
        """
        formula = list(self.defect_entries.values())[
            0
        ].defect.structure.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
        attrs = {k for k in vars(self) if not k.startswith("_")}
        methods = {k for k in dir(self) if callable(getattr(self, k)) and not k.startswith("_")}
        properties = {
            name for name, value in inspect.getmembers(type(self)) if isinstance(value, property)
        }
        return (
            f"doped DefectsSet for bulk composition {formula}, with {len(self.defect_entries)} "
            f"defect entries in self.defect_entries. Available attributes:\n{attrs | properties}\n\n"
            f"Available methods:\n{methods}"
        )


# TODO: Go through and update docstrings with descriptions all the default behaviour (INCAR,
#  KPOINTS settings etc)
# TODO: Ensure json serializability, and have optional parameter to output DefectRelaxSet jsons to
#  written folders as well (but off by default)
# TODO: Likewise, add same to/from json etc. functions for DefectRelaxSet. __Dict__ methods apply
#  to `.defect_sets` etc?
# TODO: Implement renaming folders like SnB if we try to write a folder that already exists,
#  and the structures don't match (otherwise overwrite)
