"""
Code to generate VASP defect calculation input files.
"""

import contextlib
import copy
import json
import os
import warnings
from functools import lru_cache
from importlib import resources
from typing import cast

import numpy as np
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import dumpfn, loadfn
from pymatgen.core import SETTINGS
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import BadIncarWarning, Kpoints, Poscar, Potcar
from pymatgen.io.vasp.sets import VaspInputSet
from pymatgen.util.typing import PathLike
from tqdm import tqdm

from doped import _doped_obj_properties_methods, _ignore_pmg_warnings, get_mp_context, pool_manager
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
    _get_defect_supercell_frac_coords,
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
default_potcar_dict = loadfn(os.path.join(MODULE_DIR, "VASP_sets/PotcarSet.yaml"))["POTCAR"]
default_relax_set = loadfn(os.path.join(MODULE_DIR, "VASP_sets/RelaxSet.yaml"))
default_HSE_set = loadfn(os.path.join(MODULE_DIR, "VASP_sets/HSESet.yaml"))
default_defect_set = loadfn(os.path.join(MODULE_DIR, "VASP_sets/DefectSet.yaml"))
default_defect_relax_set = copy.deepcopy(default_relax_set)
default_defect_relax_set = deep_dict_update(
    default_defect_relax_set, default_defect_set
)  # defect set is just INCAR settings
_ = default_defect_relax_set["INCAR"].pop("EDIFF_PER_ATOM")  # remove EDIFF_PER_ATOM and use defect EDIFF
singleshot_incar_settings = {
    "EDIFF": 1e-6,  # tight EDIFF for final energy and converged DOS
    "EDIFFG": None,  # no ionic relaxation, remove to avoid confusion
    "IBRION": -1,  # no ionic relaxation
    "NSW": 0,  # no ionic relaxation
    "POTIM": None,  # no ionic relaxation, remove to avoid confusion
}


def _test_potcar_functional_choice(potcar_functional: str = "PBE", symbols: list | None = None):
    """
    Check if the potcar functional choice needs to be changed to match those
    available.
    """
    test_potcar = None
    if symbols is None:
        symbols = ["Mg"]
    try:
        test_potcar = _get_potcar(tuple(symbols), potcar_functional=potcar_functional)
    except (OSError, RuntimeError) as e:  # updated to RuntimeError in pymatgen 2024.5.1
        if not potcar_functional.startswith("PBE"):
            raise e

        test_pbe_potcar_strings = ["PBE", "PBE_52", "PBE_54", "PBE_64", potcar_functional]
        # user potcar functional tested last so error message matches this
        for pbe_potcar_string in test_pbe_potcar_strings:  # try other functional choices
            with contextlib.suppress(OSError, RuntimeError):
                potcar_functional = pbe_potcar_string
                test_potcar = _get_potcar(tuple(symbols), potcar_functional=potcar_functional)
                break

        if test_potcar is None:
            # issue might be with just one of the symbols, so loop through and find the first one that
            # breaks for all, to ensure informative error message
            for symbol in symbols:
                for i, pbe_potcar_string in enumerate(test_pbe_potcar_strings):
                    try:
                        potcar_functional = pbe_potcar_string
                        test_potcar = _get_potcar((symbol,), potcar_functional=potcar_functional)
                        break
                    except (OSError, RuntimeError) as single_pot_exc:
                        if i + 1 == len(test_pbe_potcar_strings):  # failed with all
                            raise single_pot_exc

            raise e  # should already have been raised at this point

    return potcar_functional


@lru_cache(maxsize=1000)  # cache POTCAR generation to speed up generation and writing
def _get_potcar(potcar_symbols, potcar_functional) -> Potcar:
    return Potcar(list(potcar_symbols), functional=potcar_functional)


class DopedKpoints(Kpoints):
    """
    Custom implementation of ``Kpoints`` to handle encoding errors that can
    happen on some old HPCs/Linux systems.

    If an encoding error occurs upon file writing, then changes Γ to Gamma and
    Å to Angstrom in the ``KPOINTS`` comment.
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


class DopedDictSet(VaspInputSet):
    """
    Modified version of ``pymatgen`` ``VaspInputSet``, to have more robust
    ``POTCAR`` handling, expedited I/O (particularly for ``POTCAR`` generation,
    which can be slow when generating many folders), ensure ``POSCAR`` atom
    sorting, avoid encoding issues with ``KPOINTS`` comments etc.
    """

    def __init__(
        self,
        structure: Structure,
        user_incar_settings: dict | None = None,
        user_kpoints_settings: dict | Kpoints | None = None,
        user_potcar_functional: str = "PBE",
        user_potcar_settings: dict | None = None,
        auto_kpar: bool = True,
        poscar_comment: str | None = None,
        **kwargs,
    ):
        r"""
        Args:
            structure (Structure):
                ``pymatgen`` ``Structure`` object for the input structure file.
            user_incar_settings (dict):
                Dictionary of user INCAR settings (AEXX, NCORE etc.) to
                override default ``INCAR`` settings. Note that any flags that
                aren't numbers or ``True/False`` need to be input as strings
                with quotation marks (e.g. ``{"ALGO": "All"}``).
                (default: None)
            user_kpoints_settings (dict or Kpoints):
                Dictionary of user ``KPOINTS`` settings (in ``pymatgen``
                ``VaspInputSet`` format) e.g., ``{"reciprocal_density": 123}``,
                or a ``Kpoints`` object. Default is Gamma-only.
            user_potcar_functional (str):
                ``POTCAR`` functional to use. Default is "PBE" and if this
                fails, tries "PBE_52", then "PBE_54".
            user_potcar_settings (dict):
                Override the default ``POTCAR``\s, e.g. {"Li": "Li_sv"}. See
                ``doped/VASP_sets/PotcarSet.yaml`` for the default ``POTCAR``
                set.
            auto_kpar (bool):
                If ``True``, and ``KPAR`` is not set in
                ``user_incar_settings``, attempts to set ``KPAR`` to a
                reasonable value based on the k-point grid. Specifically, sets
                ``KPAR`` to 2 if there are 2 or >=4 k-points in any direction,
                or 4 if there are at least 2 directions with 2 or >=4 k-points
                (otherwise remains as the default of ``1``).
                Default is ``True``.
            poscar_comment (str):
                Comment line to use for ``POSCAR`` file. Default is structure
                formula.
            **kwargs:
                Additional kwargs to pass to ``VaspInputSet``.
        """
        _ignore_pmg_warnings()
        self.auto_kpar = auto_kpar
        self.poscar_comment = poscar_comment or structure.formula

        if user_incar_settings is not None:
            if "EDIFF_PER_ATOM" in user_incar_settings:
                ediff_per_atom = user_incar_settings.pop("EDIFF_PER_ATOM")  # pop un-used tag
                ediff = scaled_ediff(
                    len(structure),
                    ediff_per_atom=ediff_per_atom,
                    max_ediff=np.inf,  # use whatever user has set
                )
                if ediff > 1e-3:
                    warnings.warn(
                        f"EDIFF_PER_ATOM was set to {ediff_per_atom:.2e} eV/atom, which gives an "
                        f"EDIFF of {ediff:.2e} eV here. This is a very large EDIFF for VASP, and "
                        f"may cause convergence issues. Please check your INCAR settings.",
                        BadIncarWarning,
                    )
                if "EDIFF" in user_incar_settings:
                    warnings.warn(
                        "EDIFF_PER_ATOM and EDIFF both set in user_incar_settings. "
                        "EDIFF_PER_ATOM will be used.",
                        BadIncarWarning,
                    )
                user_incar_settings["EDIFF"] = ediff

            # Load INCAR tag/value check reference file from pymatgen.io.vasp.inputs
            with open(f"{resources.files('pymatgen.io.vasp')}/incar_parameters.json") as json_file:
                incar_params = json.load(json_file)

            for k in user_incar_settings:
                # check INCAR flags and warn if they don't exist (typos)
                if k not in incar_params and "#" not in k:
                    warnings.warn(  # but only checking keys, not values so we can add comments etc
                        f"Cannot find {k} from your user_incar_settings in the list of INCAR flags",
                        BadIncarWarning,
                    )

        potcar_settings = copy.deepcopy(default_potcar_dict)

        # check POTCAR settings not in config dict format:
        if isinstance(user_potcar_settings, dict):
            if "POTCAR_FUNCTIONAL" in user_potcar_settings and user_potcar_functional == "PBE":
                # i.e. default
                user_potcar_functional = user_potcar_settings.pop("POTCAR_FUNCTIONAL")
            if "POTCAR" in user_potcar_settings:
                user_potcar_settings = user_potcar_settings["POTCAR"]

        potcar_settings.update(user_potcar_settings or {})
        base_config_dict = {"POTCAR": potcar_settings}  # needs to be set with ``VaspInputSet``
        config_dict = deep_dict_update(base_config_dict, kwargs.pop("config_dict", {}))
        config_dict["INCAR"] = user_incar_settings or {}

        super().__init__(
            structure,
            config_dict=config_dict,
            user_kpoints_settings=user_kpoints_settings or {},
            user_potcar_functional=user_potcar_functional,
            force_gamma=kwargs.pop("force_gamma", True),  # force gamma-centred k-points by default
            **kwargs,
        )

    @property
    def incar(self):
        """
        Returns the ``Incar`` object generated from the ``VaspInputSet``
        config, with a warning if ``KPAR > 1`` and only one k-point.
        """
        incar_obj = super().incar

        if "KPAR" not in incar_obj and self.auto_kpar:  # determine appropriate KPAR setting
            if len(self.kpoints.kpts[0]):  # k-point mesh
                num_kpts_2_or_4_or_more = sum(i == 2 or i >= 4 for i in self.kpoints.kpts[0])
                if num_kpts_2_or_4_or_more == 1:
                    incar_obj["KPAR"] = "2  # 2 or >=4 k-points in one direction"
                elif num_kpts_2_or_4_or_more >= 2:
                    incar_obj["KPAR"] = "4  # 2 or >=4 k-points in at least two directions"

        elif "KPAR" in incar_obj and np.prod(self.kpoints.kpts[0]) == 1:
            # check KPAR setting is reasonable for number of KPOINTS
            warnings.warn("KPOINTS are Γ-only (i.e. only one kpoint), so KPAR is being set to 1")
            incar_obj["KPAR"] = "1  # Only one k-point (Γ-only)"

        return incar_obj

    @property
    def potcar(self) -> Potcar:
        """
        ``Potcar`` object.

        Redefined to intelligently handle ``pymatgen`` ``POTCAR`` issues.
        """
        if any("VASP_PSP_DIR" in i for i in SETTINGS):
            self.user_potcar_functional: str = _test_potcar_functional_choice(
                self.user_potcar_functional, self.potcar_symbols
            )

        # use our own POTCAR generation function to expedite generation and writing
        return _get_potcar(tuple(self.potcar_symbols), self.user_potcar_functional)

    @property
    def poscar(self) -> Poscar:
        """
        Return ``Poscar`` object with comment, ensuring atom sorting.
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
        Return ``kpoints`` object with comment.
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

    def __repr__(self):
        """
        Returns a string representation of the ``DopedDictSet`` object.
        """
        properties, methods = _doped_obj_properties_methods(self)
        return (
            f"doped DopedDictSet with structure composition {self.structure.composition}. "
            f"Available attributes:\n{properties}\n\nAvailable methods:\n{methods}"
        )


class DefectDictSet(DopedDictSet):
    """
    Extension to ``pymatgen`` ``VaspInputSet`` object for ``VASP`` defect
    calculations.
    """

    def __init__(
        self,
        structure: Structure,
        charge_state: int = 0,
        user_incar_settings: dict | None = None,
        user_kpoints_settings: dict | Kpoints | None = None,
        user_potcar_functional: str = "PBE",
        user_potcar_settings: dict | None = None,
        poscar_comment: str | None = None,
        **kwargs,
    ):
        r"""
        Args:
            structure (Structure):
                ``pymatgen`` ``Structure`` object of the defect supercell.
            charge_state (int):
                Charge of the defect (to set ``NELECT`` -- total number of
                electrons).
            user_incar_settings (dict):
                Dictionary of user ``INCAR`` settings (AEXX, NCORE etc.) to
                override default settings. Highly recommended to look at output
                ``INCAR``\s or the ``RelaxSet.yaml`` and ``DefectSet.yaml``
                files in the ``doped/VASP_sets`` folder, to see what the
                default ``INCAR`` settings are. Note that any flags that aren't
                numbers or ``True/False`` need to be input as strings with
                quotation marks (e.g. ``{"ALGO": "All"}``).
                (default: None)
            user_kpoints_settings (dict or Kpoints):
                Dictionary of user ``KPOINTS`` settings (in ``pymatgen``
                ``VaspInputSet`` format) e.g., ``{"reciprocal_density": 123}``,
                or a ``Kpoints`` object. Default is Gamma-centred,
                ``reciprocal_density = 100`` [Å⁻³].
            user_potcar_functional (str):
                ``POTCAR`` functional to use. Default is "PBE" and if this
                fails, tries "PBE_52", then "PBE_54".
            user_potcar_settings (dict):
                Override the default ``POTCAR``\s, e.g. ``{"Li": "Li_sv"}``.
                See ``doped/VASP_sets/PotcarSet.yaml`` for the default
                ``POTCAR`` set.
            poscar_comment (str):
                Comment line to use for ``POSCAR`` files. Default is defect
                name, fractional coordinates of initial site and charge state.
            **kwargs:
                Additional kwargs to pass to ``VaspInputSet``.
        """
        _ignore_pmg_warnings()
        self.charge_state = charge_state
        self.poscar_comment = (
            poscar_comment
            or f"{structure.formula} {'+' if self.charge_state > 0 else ''}{self.charge_state}"
        )
        custom_user_incar_settings = user_incar_settings or {}

        # get base config and set EDIFF
        relax_set = copy.deepcopy(default_defect_relax_set)

        lhfcalc = (
            True if user_incar_settings is None else user_incar_settings.get("LHFCALC", True)
        )  # True (hybrid) by default
        if lhfcalc or (isinstance(lhfcalc, str) and lhfcalc.lower().startswith("t")):
            relax_set = deep_dict_update(relax_set, default_HSE_set)  # HSE set is just INCAR settings

        input_user_incar_settings = user_incar_settings or {}
        relax_set["INCAR"].update(input_user_incar_settings)
        if "EDIFF_PER_ATOM" in input_user_incar_settings:
            relax_set["INCAR"].pop("EDIFF")  # remove base EDIFF setting and use input EDIFF_PER_ATOM

        # if "length" in user kpoint settings then pop reciprocal_density and use length instead
        if user_kpoints_settings is not None and (
            "length" in user_kpoints_settings
            if isinstance(user_kpoints_settings, dict)
            else "length" in user_kpoints_settings.as_dict()
        ):
            relax_set["KPOINTS"].pop("reciprocal_density")

        super(self.__class__, self).__init__(
            structure,
            user_incar_settings=relax_set["INCAR"],
            user_kpoints_settings=user_kpoints_settings or relax_set["KPOINTS"] or {},
            user_potcar_functional=user_potcar_functional,
            user_potcar_settings=user_potcar_settings,
            force_gamma=kwargs.pop("force_gamma", True),  # force gamma-centred k-points by default
            poscar_comment=self.poscar_comment,
            **kwargs,
        )
        self.user_incar_settings = custom_user_incar_settings
        self.user_potcar_settings = user_potcar_settings

    @property
    def incar(self):
        """
        Returns the ``Incar`` object generated from ``DopedDictSet``, with
        ``NELECT`` and ``NUPDOWN`` set accordingly.

        See https://doped.readthedocs.io/en/latest/Tips.html#spin
        for discussion about appropriate ``NUPDOWN``/``MAGMOM`` settings.
        """
        incar_obj = super(self.__class__, self).incar

        try:
            # getting NELECT can take time with many file IO calls, so only call once
            nelect = incar_obj.get("NELECT", self.nelect)
            if nelect % 2 != 0:  # odd number of electrons
                incar_obj["NUPDOWN"] = 1
            else:
                nup_0_str = "0  # see https://doped.readthedocs.io/en/latest/Tips.html#spin"
                incar_obj["NUPDOWN"] = nup_0_str  # just set to 0 upon file writing by pymatgen anyway

            if self.charge_state != 0:  # only set NELECT in the INCAR if non-neutral (easier copying of
                incar_obj["NELECT"] = nelect  # INCARs for different supercell sizes etc)

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
                f"unphysical solutions (see https://doped.readthedocs.io/en/latest/Tips.html#spin), and "
                f"POTCARs are also needed to set the charge state (i.e. NELECT) of charged defect "
                f"supercells. Got error:\n{e!r}"
            )

        return incar_obj

    @property
    def nelect(self):
        r"""
        Number of electrons (``NELECT``) for the given structure and charge
        state.

        This is equal to the sum of valence electrons (``ZVAL``) of the
        ``POTCAR``\s for each atom in the structure (supercell), minus the
        charge state.
        """
        neutral_nelect = super().nelect
        return neutral_nelect - self.charge_state

    def _check_user_potcars(self, poscar: bool = False, snb: bool = False) -> bool:
        r"""
        Check and warn the user if ``POTCAR``\s are not set up with
        ``pymatgen``.
        """
        potcars = any("VASP_PSP_DIR" in i for i in SETTINGS)
        if not potcars:
            potcar_warning_string = (
                "POTCAR directory not set up with pymatgen (see the doped docs Installation page: "
                "https://doped.readthedocs.io/en/latest/Installation.html for instructions on setting "
                "this up). This is required to generate `POTCAR` files and set the `NELECT` and "
                "`NUPDOWN` `INCAR` tags"
            )
            if poscar:
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
        output_path: PathLike,
        poscar: bool = True,
        rattle: bool = False,
        make_dir_if_not_present: bool = True,
        include_cif: bool = False,
        potcar_spec: bool = False,
        zip_output: bool = False,
        snb: bool = False,
        stdev: float | None = None,
        d_min: float | None = None,
    ):
        """
        Writes out all input to a directory.

        Refactored slightly from ``pymatgen`` ``VaspInputSet.write_input()`` to
        allow checking of user ``POTCAR`` setup, and generation of rattled
        structures.

        Args:
            output_path (PathLike):
                Directory to output the ``VASP`` input files.
            poscar (bool):
                If True, write the ``POSCAR`` to the generated folder as well.
                If ``rattle=True``, this will be a rattled structure, otherwise
                the unperturbed defect structure. (default: True)
            rattle (bool):
                If writing ``POSCAR``, apply random displacements to all atomic
                positions in the structure using the ``ShakeNBreak`` algorithm;
                i.e. with the displacement distances randomly drawn from a
                Gaussian distribution of standard deviation equal to 10% of the
                nearest neighbour distance and using a Monte Carlo algorithm to
                penalise displacements that bring atoms closer than 80% of the
                nearest neighbour distance.
                ``stdev`` and ``d_min`` can also be given as input kwargs.
                This is intended to be used as a fallback option for breaking
                symmetry to partially aid location of global minimum defect
                geometries, if ``ShakeNBreak`` structure-searching is being
                skipped. However, rattling still only finds the ground-state
                structure for <~30% of known cases of energy-lowering
                reconstructions relative to an unperturbed defect structure.
                (default: False)
            make_dir_if_not_present (bool):
                Set to ``True`` if you want the directory (and the whole path)
                to be created if it is not present. (default: True)
            include_cif (bool):
                Whether to write a CIF file in the output
                directory for easier opening by VESTA. (default: False)
            potcar_spec (bool):
                Instead of writing the POTCAR, write a "POTCAR.spec".
                This is intended to help sharing an input set with people who
                might not have a license to specific Potcar files. Given a
                ``POTCAR.spec`` file, the specific POTCAR file can be
                re-generated using ``pymatgen`` with the ``generate_potcar``
                function in the ``pymatgen`` CLI. (default: False)
            zip_output (bool):
                Whether to zip each VASP input file written to the output
                directory. (default: False)
            snb (bool):
                If input structures are from ``ShakeNBreak`` (so POSCARs aren't
                'unperturbed') -- only really intended for internal use by
                ``ShakeNBreak``. (default: False)
            stdev (float):
                Standard deviation for the Gaussian distribution of
                displacements for the ``ShakeNBreak`` rattling algorithm. If
                ``None`` (default) this is set to 10% of the nearest neighbour
                distance in the structure.
            d_min (float):
                Minimum interatomic distance (in Angstroms) in the rattled
                structure. Monte Carlo rattle moves that put atoms at distances
                less than this will be heavily penalised. Default is to set
                this to 80% of the nearest neighbour distance in the structure.
        """
        potcars = True if potcar_spec else self._check_user_potcars(poscar=poscar, snb=snb)

        if poscar and rattle:
            try:
                from shakenbreak.distortions import rattle as SnB_rattle
            except ImportError as e:
                raise ImportError(
                    "ShakeNBreak must be installed (pip install shakenbreak) to use the rattle option!"
                ) from e
            self.structure: Structure = SnB_rattle(self.structure, stdev=stdev, d_min=d_min)

        if poscar and potcars:  # write everything, use VaspInputSet.write_input()
            try:
                super().write_input(
                    output_path,
                    make_dir_if_not_present=make_dir_if_not_present,
                    include_cif=include_cif,
                    potcar_spec=potcar_spec,
                    zip_output=zip_output,
                )
            except ValueError as e:
                if not str(e).startswith("NELECT") or not potcar_spec:
                    raise e

                os.makedirs(output_path, exist_ok=True)
                with zopen(os.path.join(output_path, "POTCAR.spec"), "wt") as pot_spec_file:
                    pot_spec_file.write("\n".join(self.potcar_symbols))

                self.kpoints.write_file(f"{output_path}/KPOINTS")
                self.poscar.write_file(f"{output_path}/POSCAR")

        else:  # use `write_file()`s rather than `write_input()` to avoid writing POSCARs/POTCARs
            os.makedirs(output_path, exist_ok=True)

            # if not POTCARs and charge_state not 0, but ``poscar`` is true, then skip INCAR
            # write attempt (POSCARs and KPOINTS will be written, and user already warned):
            if potcars or self.charge_state == 0 or not poscar:
                self.incar.write_file(f"{output_path}/INCAR")

            if potcars:
                if potcar_spec:
                    with zopen(os.path.join(output_path, "POTCAR.spec"), "wt") as pot_spec_file:
                        pot_spec_file.write("\n".join(self.potcar_symbols))
                else:
                    self.potcar.write_file(f"{output_path}/POTCAR")

            self.kpoints.write_file(f"{output_path}/KPOINTS")

            if poscar:
                self.poscar.write_file(f"{output_path}/POSCAR")

    def __repr__(self):
        """
        Returns a string representation of the ``DefectDictSet`` object.
        """
        properties, methods = _doped_obj_properties_methods(self)
        return (
            f"doped DefectDictSet with supercell composition {self.structure.composition}. "
            f"Available attributes:\n{properties}\n\nAvailable methods:\n{methods}"
        )


def scaled_ediff(natoms: int, ediff_per_atom: float = 2e-7, max_ediff: float = 1e-4) -> float:
    """
    Returns a scaled ``EDIFF`` value for VASP calculations, based on the number
    of atoms in the structure.

    Args:
        natoms (int): Number of atoms in the structure.
        ediff_per_atom (float):
            Per-atom ``EDIFF`` in eV. Default is 2e-7 (1e-5 per 50 atoms).
        max_ediff (float): Maximum ``EDIFF`` value. Default is 1e-4 eV.

    Returns:
        float: Scaled ``EDIFF`` value.
    """
    ediff = float(f"{natoms*ediff_per_atom:.1g}")
    return min(ediff, max_ediff)


class DefectRelaxSet(MSONable):
    """
    Class for generating input files for ``VASP`` defect relaxation
    calculations for a single ``pymatgen`` ``DefectEntry`` or ``Structure``
    object.
    """

    def __init__(
        self,
        defect_entry: DefectEntry | Structure,
        charge_state: int | None = None,
        soc: bool | None = None,
        user_incar_settings: dict | None = None,
        user_kpoints_settings: dict | Kpoints | None = None,
        user_potcar_functional: str = "PBE",
        user_potcar_settings: dict | None = None,
        **kwargs,
    ):
        r"""
        The supercell structure and charge state are taken from the
        ``DefectEntry`` attributes, or if a ``Structure`` is provided, then
        from the ``defect_supercell`` and ``charge_state`` input parameters.

        Creates attributes:

        - ``DefectRelaxSet.vasp_gam``:
            ``DefectDictSet`` for Gamma-point only relaxation. Usually not
            needed if ``ShakeNBreak`` (or other) structure searching has been
            performed (recommended), unless only Γ-point `k`-point sampling is
            required (converged) for your system, and no ``vasp_std``
            calculations with multiple `k`-points are required (determined from
            kpoint settings).
        - ``DefectRelaxSet.vasp_nkred_std``:
            ``DefectDictSet`` for relaxation with a kpoint mesh and using
            ``NKRED``. Not generated for GGA calculations (if ``LHFCALC`` is
            set to ``False`` in ``user_incar_settings``) or if only Gamma
            `k`-point sampling is required.
        - ``DefectRelaxSet.vasp_std``:
            ``DefectDictSet`` for relaxation with a kpoint mesh, not using
            ``NKRED``. Not generated if only Gamma kpoint sampling is required.
        - ``DefectRelaxSet.vasp_ncl``:
            ``DefectDictSet`` for single-point (static) energy calculation with
            SOC included. Generated if ``soc=True``. If ``soc`` is not set,
            then by default is only generated for systems with a max atomic
            number (Z) >= 31 (i.e. further down the periodic table than Zn).

        where ``DefectDictSet`` is an extension of ``pymatgen``'s
        ``VaspInputSet`` class for defect calculations, with ``incar``,
        ``poscar``, ``kpoints`` and ``potcar`` attributes for the corresponding
        VASP defect calculations (see docstring). Also creates the
        corresponding ``bulk_vasp_...`` attributes for single-point (static)
        energy calculations of the bulk (pristine, defect-free) supercell. This
        needs to be calculated once with the same settings as the defect
        calculations, for the later calculation of defect formation energies.

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT``
        settings, and ``PotcarSet.yaml`` for the default ``POTCAR`` settings.
        **These are reasonable defaults that `roughly` match the typical values
        needed for accurate defect calculations, but usually will need to be
        modified for your specific system, such as converged ENCUT and KPOINTS,
        and NCORE / KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings
        should be consistent with those used for all defect and competing phase
        (chemical potential) calculations -- this will be automatically checked
        upon defect & competing phases parsing in ``doped``.

        Args:
            defect_entry (DefectEntry, Structure):
                ``doped``/``pymatgen`` ``DefectEntry`` or ``Structure`` (defect
                supercell) for which to generate ``DefectDictSet``\s for.
            charge_state (int):
                Charge state of the defect. Overrides
                ``DefectEntry.charge_state`` if ``DefectEntry`` is input.
            soc (bool):
                Whether to generate ``vasp_ncl`` DefectDictSet attribute for
                spin-orbit coupling single-point (static) energy calculations.
                If not set, then by default is set to True if the max atomic
                number (Z) in the structure is >= 31 (i.e. further down the
                periodic table than Zn), otherwise ``False``.
            user_incar_settings (dict):
                Dictionary of user ``INCAR`` settings (AEXX, NCORE etc.) to
                override default settings. Highly recommended to look at output
                ``INCAR``\s or the ``RelaxSet.yaml`` and ``DefectSet.yaml``
                files in the ``doped/VASP_sets`` folder, to see what the
                default ``INCAR`` settings are. Note that any flags that aren't
                numbers or ``True/False`` need to be input as strings with
                quotation marks (e.g. ``{"ALGO": "All"}``).
                (default: None)
            user_kpoints_settings (dict or Kpoints):
                Dictionary of user ``KPOINTS`` settings (in ``pymatgen``
                ``VaspInputSet`` format) e.g., ``{"reciprocal_density": 123}``,
                or a ``Kpoints`` object, to use for the ``vasp_std``,
                ``vasp_nkred_std`` and ``vasp_ncl`` ``DefectDictSet``\s (Γ-only
                for ``vasp_gam``). Default is Gamma-centred,
                ``reciprocal_density = 100`` [Å⁻³].
            user_potcar_functional (str):
                ``POTCAR`` functional to use. Default is "PBE" and if this
                fails, tries "PBE_52", then "PBE_54".
            user_potcar_settings (dict):
                Override the default ``POTCAR``\s, e.g. ``{"Li": "Li_sv"}``.
                See ``doped/VASP_sets/PotcarSet.yaml`` for the default
                ``POTCAR`` set.
            **kwargs: Additional kwargs to pass to ``DefectDictSet``.

        Key Attributes:
            vasp_gam (DefectDictSet):
                ``DefectDictSet`` for Gamma-point only relaxation. Usually not
                needed if ``ShakeNBreak`` (or other) structure searching has
                been performed (recommended), unless only Γ-point `k`-point
                sampling is required (converged) for your system, and no
                ``vasp_std`` calculations with multiple `k`-points are required
                (determined from kpoints settings).
            vasp_nkred_std (DefectDictSet):
                ``DefectDictSet`` for relaxation with a non-Γ-only kpoint mesh,
                using ``NKRED(X,Y,Z)`` INCAR tag(s) to downsample kpoints for
                the HF exchange part of the hybrid DFT calculation. Not
                generated for GGA calculations (if ``LHFCALC`` is set to
                ``False`` in ``user_incar_settings``) or if only Gamma kpoint
                sampling is required.
            vasp_std (DefectDictSet):
                ``DefectDictSet`` for relaxation with a non-Γ-only kpoint mesh,
                not using ``NKRED``. Not generated if only Gamma kpoint
                sampling is required.
            vasp_ncl (DefectDictSet):
                ``DefectDictSet`` for single-point (static) energy calculation
                with SOC included. Generated if ``soc=True``. If ``soc`` is not
                set, then by default is only generated for systems with a max
                atomic number (Z) >= 31 (i.e. further down the periodic table
                than Zn).
            defect_supercell (Structure):
                Supercell structure for defect calculations, taken from
                ``defect_entry.defect_supercell`` (if defined), otherwise from
                ``defect_entry.sc_entry.structure`` if inputting a
                ``DefectEntry`` object, or the input structure if inputting a
                ``Structure`` object.
            bulk_supercell (Structure):
                Supercell structure of the bulk (pristine, defect-free)
                material, taken from ``defect_entry.bulk_supercell`` (if
                defined), otherwise from ``defect_entry.bulk_entry.structure``
                if inputting a ``DefectEntry`` object, or ``None`` if inputting
                a ``Structure`` object.
            poscar_comment (str):
                Comment to write at the top of the ``POSCAR`` files. Default is
                the defect entry name, defect frac coords and charge state (if
                inputting a ``DefectEntry`` object), or the formula of the
                input structure and charge state (if inputting a ``Structure``
                object), for defects. For the bulk supercell, it's
                ``"{formula} - Bulk"``.
            bulk_vasp_gam (DefectDictSet):
                ``DefectDictSet`` for a `bulk` Γ-point-only single-point
                (static) supercell calculation. Often not used, as the bulk
                supercell only needs to be calculated once with the same
                settings as the final defect calculations, which may be with
                ``vasp_std`` or ``vasp_ncl``.
            bulk_vasp_nkred_std (DefectDictSet):
                ``DefectDictSet`` for a single-point (static) `bulk`
                ``vasp_std`` supercell calculation (i.e. with a non-Γ-only
                kpoint mesh) and ``NKRED(X,Y,Z)`` ``INCAR`` tag(s) to
                downsample kpoints for the HF exchange part of the hybrid DFT
                calculation. Not generated for GGA calculations (if ``LHFCALC``
                is set to ``False`` in ``user_incar_settings``) or if only
                Gamma kpoint sampling is required.
            bulk_vasp_std (DefectDictSet):
                ``DefectDictSet`` for a single-point (static) `bulk`
                ``vasp_std`` supercell calculation with a non-Γ-only kpoint
                mesh, not using ``NKRED``. Not generated if only Gamma kpoint
                sampling is required.
            bulk_vasp_ncl (DefectDictSet):
                ``DefectDictSet`` for single-point (static) energy calculation
                of the `bulk` supercell with SOC included. Generated if
                ``soc=True``. If ``soc`` is not set, then by default is only
                generated for systems with a max atomic number (Z) >= 31 (i.e.
                further down the periodic table than Zn).

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
            sc_frac_coords = _get_defect_supercell_frac_coords(self.defect_entry)
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
        Returns a ``DefectDictSet`` object for a VASP Γ-point-only
        (``vasp_gam``) defect supercell relaxation. Typically not needed if
        ShakeNBreak (or other) structure searching has been performed
        (recommended), unless only Γ-point `k`-point sampling is required
        (converged) for your system, and no vasp_std calculations with multiple
        `k`-points are required (determined from kpoints settings).

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT``
        settings, and ``PotcarSet.yaml`` for the default ``POTCAR`` settings.
        **These are reasonable defaults that `roughly` match the typical values
        needed for accurate defect calculations, but usually will need to be
        modified for your specific system, such as converged ENCUT and KPOINTS,
        and NCORE / KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings
        should be consistent with those used for all defect and competing phase
        (chemical potential) calculations -- this will be automatically checked
        upon defect & competing phases parsing in ``doped``.
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

    def _check_vstd_kpoints(
        self, vasp_std_defect_set: DefectDictSet, warn: bool = True, info: bool = True
    ) -> DefectDictSet | None:
        try:
            vasp_std = None if np.prod(vasp_std_defect_set.kpoints.kpts[0]) == 1 else vasp_std_defect_set
        except Exception:  # different kpoint generation scheme, not Gamma-only
            vasp_std = vasp_std_defect_set

        if vasp_std is None and (warn or info):
            current_kpoint_settings = (
                self.user_kpoints_settings
                or "default `reciprocal_density = 100` [Å⁻³] (see doped/VASP_sets/RelaxSet.yaml)"
            )
            info_message = (
                f"With the current kpoint settings ({current_kpoint_settings}), the k-point "
                f"mesh is Γ-only (see docstrings)."
            )
            if warn:
                warnings.warn(
                    f"{info_message} Thus vasp_std should not be used, and vasp_gam should be used "
                    f"instead."
                )
            else:
                print(info_message)

        return vasp_std

    @property
    def vasp_std(self) -> DefectDictSet | None:
        """
        Returns a ``DefectDictSet`` object for a VASP defect supercell
        relaxation using ``vasp_std`` (i.e. with a non-Γ-only kpoint mesh).
        Returns None and a warning if the input kpoint settings correspond to a
        Γ-only kpoint mesh (in which case ``vasp_gam`` should be used).

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT``
        settings, and ``PotcarSet.yaml`` for the default ``POTCAR`` settings.
        **These are reasonable defaults that `roughly` match the typical values
        needed for accurate defect calculations, but usually will need to be
        modified for your specific system, such as converged ENCUT and KPOINTS,
        and NCORE / KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings
        should be consistent with those used for all defect and competing phase
        (chemical potential) calculations -- this will be automatically checked
        upon defect & competing phases parsing in ``doped``.
        """
        # determine if vasp_std required or only vasp_gam:
        return self._check_vstd_kpoints(self._vasp_std)

    @property
    def _vasp_std(self) -> DefectDictSet:
        return DefectDictSet(
            self.defect_supercell,
            charge_state=self.defect_entry.charge_state,
            user_incar_settings=self.user_incar_settings,
            user_kpoints_settings=self.user_kpoints_settings,
            user_potcar_functional=self.user_potcar_functional,
            user_potcar_settings=self.user_potcar_settings,
            poscar_comment=self.poscar_comment,
            **self.dict_set_kwargs,
        )

    @property
    def vasp_nkred_std(self) -> DefectDictSet | None:
        """
        Returns a ``DefectDictSet`` object for a VASP defect supercell
        relaxation using ``vasp_std`` (i.e. with a non-Γ-only kpoint mesh) and
        ``NKRED(X,Y,Z)`` INCAR tag(s) to downsample kpoints for the HF exchange
        part of hybrid DFT calculations, following the doped recommended defect
        calculation workflow (see docs). By default, sets ``NKRED(X,Y,Z)`` to 2
        or 3 in the directions for which the k-point grid is divisible by this
        factor. Returns None and a warning if the input kpoint settings
        correspond to a Γ-only kpoint mesh (in which case ``vasp_gam`` should
        be used) or for GGA calculations (if ``LHFCALC`` is set to ``False`` in
        user_incar_settings, in which case ``vasp_std`` should be used).

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT``
        settings, and ``PotcarSet.yaml`` for the default ``POTCAR`` settings.
        **These are reasonable defaults that `roughly` match the typical values
        needed for accurate defect calculations, but usually will need to be
        modified for your specific system, such as converged ENCUT and KPOINTS,
        and NCORE / KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings
        should be consistent with those used for all defect and competing phase
        (chemical potential) calculations -- this will be automatically checked
        upon defect & competing phases parsing in ``doped``.
        """
        if self.user_incar_settings.get("LHFCALC", True) is False:  # GGA
            return None

        std_defect_set = self.vasp_std  # warns the user if Γ-only

        return self._vasp_nkred_std if std_defect_set is not None else None  # non Γ-only kpoint mesh?

    @property
    def _vasp_nkred_std(self):
        std_defect_set = self._vasp_std
        # determine appropriate NKRED:
        try:
            kpt_mesh = np.array(std_defect_set.kpoints.kpts[0])
            # if all divisible by 2 or 3, then set NKRED to 2 or 3, respectively:
            nkred_dict = {
                "NKRED": None,
                "NKREDX": None,
                "NKREDY": None,
                "NKREDZ": None,
            }  # type: dict[str, Optional[int]]
            for k in [2, 3]:
                if np.all(kpt_mesh % k == 0):
                    nkred_dict["NKRED"] = k
                    break

                for idx, nkred_key in enumerate(["NKREDX", "NKREDY", "NKREDZ"]):
                    if kpt_mesh[idx] % k == 0:
                        nkred_dict[nkred_key] = k
                        break

            incar_settings = copy.deepcopy(self.user_incar_settings)
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

    @property
    def vasp_ncl(self) -> DefectDictSet | None:
        """
        Returns a ``DefectDictSet`` object for a VASP defect supercell single-
        point calculation with spin-orbit coupling (SOC) included (i.e.
        ``LSORBIT = True``), using ``vasp_ncl``. If ``DefectRelaxSet.soc`` is
        False, then this returns None and a warning. If the ``soc`` parameter
        is not set when initializing ``DefectRelaxSet``, then this is set to
        True for systems with a max atomic number (Z) >= 31 (i.e. further down
        the periodic table than Zn), otherwise False.

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT``
        settings, and ``PotcarSet.yaml`` for the default ``POTCAR`` settings.
        **These are reasonable defaults that `roughly` match the typical values
        needed for accurate defect calculations, but usually will need to be
        modified for your specific system, such as converged ENCUT and KPOINTS,
        and NCORE / KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings
        should be consistent with those used for all defect and competing phase
        (chemical potential) calculations -- this will be automatically checked
        upon defect & competing phases parsing in ``doped``.
        """
        if not self.soc:
            return None

        user_incar_settings = copy.deepcopy(self.user_incar_settings)
        user_incar_settings.update(singleshot_incar_settings)
        user_incar_settings.update({"LSORBIT": True})  # ISYM already 0

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
    def bulk_vasp_gam(self) -> DefectDictSet | None:
        """
        Returns a ``DefectDictSet`` object for a VASP `bulk` Γ-point-only
        (``vasp_gam``) single-point (static) supercell calculation. Often not
        used, as the bulk supercell only needs to be calculated once with the
        same settings as the final defect calculations, which is ``vasp_std``
        if we have a non-Γ-only final k-point mesh, or ``vasp_ncl`` if SOC
        effects are being included. If the final converged k-point mesh is
        Γ-only, then ``bulk_vasp_gam`` should be used to calculate the single-
        point (static) bulk supercell reference energy. Can also sometimes be
        useful for the purpose of calculating defect formation energies at
        early stages of the typical ``vasp_gam`` -> ``vasp_nkred_std`` (if.

        hybrid & non-Γ-only k-points) -> ``vasp_std`` (if non-Γ-only k-points)
        -> ``vasp_ncl`` (if SOC included) workflow, to obtain rough formation
        energy estimates and flag any potential issues with defect calculations
        early on.

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT``
        settings, and ``PotcarSet.yaml`` for the default ``POTCAR`` settings.
        **These are reasonable defaults that `roughly` match the typical values
        needed for accurate defect calculations, but usually will need to be
        modified for your specific system, such as converged ENCUT and KPOINTS,
        and NCORE / KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings
        should be consistent with those used for all defect and competing phase
        (chemical potential) calculations -- this will be automatically checked
        upon defect & competing phases parsing in ``doped``.
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
            poscar_comment=f"{bulk_supercell.formula} -- Bulk",
            **self.dict_set_kwargs,
        )

    @property
    def bulk_vasp_std(self) -> DefectDictSet | None:
        """
        Returns a ``DefectDictSet`` object for a single-point (static) `bulk`
        ``vasp_std`` supercell calculation. Returns None and a warning if the
        input kpoint settings correspond to a Γ-only kpoint mesh (in which case
        ``(bulk_)vasp_gam`` should be used).

        The bulk supercell only needs to be calculated once with the same
        settings as the final defect calculations, which is ``vasp_std`` if we
        have a non-Γ-only final k-point mesh, ``vasp_ncl`` if SOC effects are
        being included (in which case ``bulk_vasp_ncl`` should be used for the
        single-point bulk supercell reference calculation), or ``vasp_gam`` if
        the final converged k-point mesh is Γ-only (in which case
        ``bulk_vasp_gam`` should be used for the single-point bulk supercell
        reference calculation). Can also sometimes be useful for the purpose of
        calculating defect formation energies at midway stages of the typical
        ``vasp_gam`` -> ``vasp_nkred_std`` (if hybrid & non-Γ-only k-points) ->
        ``vasp_std`` (if non-Γ-only k-points) -> ``vasp_ncl`` (if SOC included)
        workflow, to obtain rough formation energy estimates and flag any
        potential issues with defect calculations early on.

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT``
        settings, and ``PotcarSet.yaml`` for the default ``POTCAR`` settings.
        **These are reasonable defaults that `roughly` match the typical values
        needed for accurate defect calculations, but usually will need to be
        modified for your specific system, such as converged ENCUT and KPOINTS,
        and NCORE / KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings
        should be consistent with those used for all defect and competing phase
        (chemical potential) calculations -- this will be automatically checked
        upon defect & competing phases parsing in ``doped``.
        """
        bulk_supercell: Structure = self._check_bulk_supercell_and_warn()
        if bulk_supercell is None:
            return None

        user_incar_settings = copy.deepcopy(self.user_incar_settings)
        user_incar_settings.update(singleshot_incar_settings)

        return self._check_vstd_kpoints(
            DefectDictSet(
                bulk_supercell,
                charge_state=0,
                user_incar_settings=user_incar_settings,
                user_kpoints_settings=self.user_kpoints_settings,
                user_potcar_functional=self.user_potcar_functional,
                user_potcar_settings=self.user_potcar_settings,
                poscar_comment=f"{bulk_supercell.formula} -- Bulk",
                **self.dict_set_kwargs,
            )
        )

    @property
    def bulk_vasp_nkred_std(self) -> DefectDictSet | None:
        """
        Returns a ``DefectDictSet`` object for a single-point (static) `bulk`
        ``vasp_std`` supercell calculation (i.e. with a non-Γ-only kpoint mesh)
        and ``NKRED(X,Y,Z)`` INCAR tag(s) to downsample kpoints for the HF
        exchange part of the hybrid DFT calculation. By default, sets
        ``NKRED(X,Y,Z)`` to 2 or 3 in the directions for which the k-point grid
        is divisible by this factor. Returns None and a warning if the input
        kpoint settings correspond to a Γ-only kpoint mesh (in which case
        ``(bulk_)vasp_gam`` should be used) or for GGA calculations (if
        ``LHFCALC`` is set to ``False`` in user_incar_settings, in which case
        ``(bulk_)vasp_std`` should be used).

        The bulk supercell only needs to be calculated once with the same
        settings as the final defect calculations, which is ``vasp_std`` if we
        have a non-Γ-only final k-point mesh, ``vasp_ncl`` if SOC effects are
        being included (in which case ``bulk_vasp_ncl`` should be used for the
        single-point bulk supercell reference calculation), or ``vasp_gam`` if
        the final converged k-point mesh is Γ-only (in which case
        ``bulk_vasp_gam`` should be used for the single-point bulk supercell
        reference calculation). Can also sometimes be useful for the purpose of
        calculating defect formation energies at midway stages of the typical
        ``vasp_gam`` -> ``vasp_nkred_std`` (if hybrid & non-Γ-only k-points) ->
        ``vasp_std`` (if non-Γ-only k-points) -> ``vasp_ncl`` (if SOC included)
        workflow, to obtain rough formation energy estimates and flag any
        potential issues with defect calculations early on.

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT``
        settings, and ``PotcarSet.yaml`` for the default ``POTCAR`` settings.
        **These are reasonable defaults that `roughly` match the typical values
        needed for accurate defect calculations, but usually will need to be
        modified for your specific system, such as converged ENCUT and KPOINTS,
        and NCORE / KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings
        should be consistent with those used for all defect and competing phase
        (chemical potential) calculations -- this will be automatically checked
        upon defect & competing phases parsing in ``doped``.
        """
        bulk_supercell: Structure = self._check_bulk_supercell_and_warn()
        if bulk_supercell is None:
            return None

        # check NKRED by running through ``vasp_nkred_std``:
        nkred_defect_dict_set = self.vasp_nkred_std

        if nkred_defect_dict_set is None:
            return None

        if any("VASP_PSP_DIR" in i for i in SETTINGS):  # POTCARs available
            user_incar_settings = copy.deepcopy(self.user_incar_settings)
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
            poscar_comment=f"{bulk_supercell.formula} -- Bulk",
            **self.dict_set_kwargs,
        )

    @property
    def bulk_vasp_ncl(self) -> DefectDictSet | None:
        """
        Returns a ``DefectDictSet`` object for VASP `bulk` supercell single-
        point calculations with spin-orbit coupling (SOC) included (i.e.
        ``LSORBIT = True``), using ``vasp_ncl``. If ``DefectRelaxSet.soc`` is
        False, then this returns None and a warning. If the ``soc`` parameter
        is not set when initializing ``DefectRelaxSet``, then this is set to
        True for systems with a max atomic number (Z) >= 31 (i.e. further down
        the periodic table than Zn), otherwise False.

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT``
        settings, and ``PotcarSet.yaml`` for the default ``POTCAR`` settings.
        **These are reasonable defaults that `roughly` match the typical values
        needed for accurate defect calculations, but usually will need to be
        modified for your specific system, such as converged ENCUT and KPOINTS,
        and NCORE / KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings
        should be consistent with those used for all defect and competing phase
        (chemical potential) calculations -- this will be automatically checked
        upon defect & competing phases parsing in ``doped``.
        """
        bulk_supercell: Structure = self._check_bulk_supercell_and_warn()
        if bulk_supercell is None:
            return None

        if not self.soc:
            return None

        user_incar_settings = copy.deepcopy(self.user_incar_settings)
        user_incar_settings.update(singleshot_incar_settings)
        user_incar_settings.update({"LSORBIT": True})  # ISYM already 0

        return DefectDictSet(
            bulk_supercell,
            charge_state=0,
            user_incar_settings=user_incar_settings,
            user_kpoints_settings=self.user_kpoints_settings,
            user_potcar_functional=self.user_potcar_functional,
            user_potcar_settings=self.user_potcar_settings,
            poscar_comment=f"{bulk_supercell.formula} -- Bulk",
            **self.dict_set_kwargs,
        )

    def _get_output_path(self, defect_dir: PathLike | None = None, subfolder: PathLike | None = None):
        if defect_dir is None:
            if self.defect_entry.name is None:
                self.defect_entry.name = get_defect_name_from_entry(self.defect_entry, relaxed=False)

            defect_dir = self.defect_entry.name

        return f"{defect_dir}/{subfolder}" if subfolder is not None else defect_dir

    def _write_vasp_xxx_files(self, defect_dir, subfolder, poscar, rattle, vasp_xxx_attribute, **kwargs):
        output_path = self._get_output_path(defect_dir, subfolder)

        stdev = d_min = None
        if rattle:
            for trial_structure in [
                self.defect_entry.defect.structure,
                self.defect_entry.bulk_supercell,
                self.defect_entry.bulk_supercell * 2,
            ]:
                distance_matrix = trial_structure.distance_matrix
                sorted_distances = np.sort(distance_matrix[distance_matrix > 0.8].flatten())
                if len(sorted_distances) > 0:
                    stdev = 0.1 * sorted_distances[0]
                    d_min = 0.8 * sorted_distances[0]
                    break

        vasp_xxx_attribute.write_input(
            output_path,
            poscar,
            rattle=rattle,
            stdev=stdev,
            d_min=d_min,
            **kwargs,  # kwargs to allow POTCAR testing on GH Actions
        )

        if "bulk" not in defect_dir:  # not a bulk supercell
            self.defect_entry.to_json(f"{output_path}/{self.defect_entry.name}.json.gz")

    def write_gam(
        self,
        defect_dir: PathLike | None = None,
        subfolder: PathLike | None = "vasp_gam",
        poscar: bool = True,
        rattle: bool = True,
        bulk: bool = False,
        **kwargs,
    ):
        r"""
        Write the input files for VASP Γ-point-only (``vasp_gam``) defect
        supercell relaxation. Typically not recommended for use, as the
        recommended workflow is to perform ``vasp_gam`` calculations using
        ``ShakeNBreak`` (or other approaches) for defect structure-searching
        and initial relaxations, but should be used if the final, converged
        `k`-point mesh is Γ-point-only. If ``bulk`` is True, the input files
        for a single-point calculation of the bulk supercell are also written
        to ``"{formula}_bulk/{subfolder}"``.

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT``
        settings, and ``PotcarSet.yaml`` for the default ``POTCAR`` settings.
        **These are reasonable defaults that `roughly` match the typical values
        needed for accurate defect calculations, but usually will need to be
        modified for your specific system, such as converged ENCUT and KPOINTS,
        and NCORE / KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings
        should be consistent with those used for all defect and competing phase
        (chemical potential) calculations -- this will be automatically checked
        upon defect & competing phases parsing in ``doped``.

        The ``DefectEntry`` object is also written to a ``json.gz`` file in
        ``defect_dir`` to aid calculation provenance -- can be reloaded
        directly with ``loadfn()`` from ``monty.serialization``, or
        ``DefectEntry.from_json()``.

        Args:
            defect_dir (PathLike):
                Folder in which to create the ``VASP`` defect calculation
                inputs. Default is to use the ``DefectEntry`` name (e.g.
                ``"Y_i_C4v_O1.92_+2"`` etc.), from ``self.defect_entry.name``.
                If this attribute is not set, it is automatically generated
                according to the doped convention (using
                ``get_defect_name_from_entry()``).
            subfolder (PathLike):
                Output folder structure is ``<defect_dir>/<subfolder>`` where
                ``subfolder`` = 'vasp_gam' by default. Setting ``subfolder`` to
                ``None`` will write the ``vasp_gam`` input files directly to
                the ``<defect_dir>`` folder, with no subfolders created.
            poscar (bool):
                If ``True`` (default), writes the defect ``POSCAR`` to the
                generated folder as well. Typically not recommended, as the
                recommended workflow is to initially perform ``vasp_gam``
                ground-state structure searching using ``ShakeNBreak``
                (https://shakenbreak.readthedocs.io) or another approach, then
                continue the ``vasp(_nkred)_std`` relaxations from the
                ground-state structures (e.g. using ``-d vasp_nkred_std`` with
                ``snb-groundstate`` (CLI) or
                ``groundstate_folder="vasp_nkred_std"`` with
                ``write_groundstate_structure`` (Python API)).
                (default: True)
            rattle (bool):
                If writing ``POSCAR``, apply random displacements to all atomic
                positions in the structure using the ``ShakeNBreak`` algorithm;
                i.e. with the displacement distances randomly drawn from a
                Gaussian distribution of standard deviation equal to 10% of the
                bulk nearest neighbour distance and using a Monte Carlo
                algorithm to penalise displacements that bring atoms closer
                than 80% of the bulk nearest neighbour distance.
                ``stdev`` and ``d_min`` can also be given as input kwargs.
                This is intended to be used as a fallback option for breaking
                symmetry to partially aid location of global minimum defect
                geometries, if ``ShakeNBreak`` structure-searching is being
                skipped. However, rattling still only finds the ground-state
                structure for <~30% of known cases of energy-lowering
                reconstructions relative to an unperturbed defect structure.
                (default: True)
            bulk (bool):
                If ``True``, the input files for a single-point calculation of
                the bulk supercell are also written to
                "{formula}_bulk/{subfolder}". (Default: False)
            **kwargs:
                Keyword arguments to pass to ``DefectDictSet.write_input()``.
        """
        if defect_dir is None:
            defect_dir = self.defect_entry.name

        self._write_vasp_xxx_files(
            defect_dir,
            subfolder,
            poscar=poscar,
            rattle=rattle,
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
                poscar=True,
                rattle=False,
                vasp_xxx_attribute=self.bulk_vasp_gam,
                **kwargs,
            )

    def write_std(
        self,
        defect_dir: PathLike | None = None,
        subfolder: PathLike | None = "vasp_std",
        poscar: bool = False,
        rattle: bool = True,
        bulk: bool = False,
        **kwargs,
    ):
        r"""
        Write the input files for a VASP defect supercell calculation using
        ``vasp_std`` (i.e. with a non-Γ-only kpoint mesh).

        By default, does not generate ``POSCAR`` (input structure) files, as
        these should be taken from the ``CONTCAR``\s of ``vasp_std``
        relaxations using ``NKRED(X,Y,Z)`` (originally from structure-searching
        relaxations) if using hybrid DFT, or from ``ShakeNBreak`` calculations
        (via ``snb-groundstate -d vasp_std``) if using GGA, or, if not
        following the recommended structure-searching workflow, from the
        ``CONTCAR``\s of ``vasp_gam`` calculations. If ``POSCAR`` files are
        desired, set ``poscar=True``. If ``bulk`` is ``True``, the input files
        for a single-point calculation of the bulk supercell are also written
        to "{formula}_bulk/{subfolder}".

        Returns None and a warning if the input kpoint settings correspond to a
        Γ-only kpoint mesh (in which case ``vasp_gam`` should be used).

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT``
        settings, and ``PotcarSet.yaml`` for the default ``POTCAR`` settings.
        **These are reasonable defaults that `roughly` match the typical values
        needed for accurate defect calculations, but usually will need to be
        modified for your specific system, such as converged ENCUT and KPOINTS,
        and NCORE / KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings
        should be consistent with those used for all defect and competing phase
        (chemical potential) calculations -- this will be automatically checked
        upon defect & competing phases parsing in ``doped``.

        The ``DefectEntry`` object is also written to a ``json.gz`` file in
        ``defect_dir`` to aid calculation provenance -- can be reloaded
        directly with ``loadfn()`` from ``monty.serialization``, or
        ``DefectEntry.from_json()``.

        Args:
            defect_dir (PathLike):
                Folder in which to create the ``VASP`` defect calculation
                inputs. Default is to use the ``DefectEntry`` name (e.g.
                ``"Y_i_C4v_O1.92_+2"`` etc.), from ``self.defect_entry.name``.
                If this attribute is not set, it is automatically generated
                according to the doped convention (using
                ``get_defect_name_from_entry()``).
            subfolder (PathLike):
                Output folder structure is ``<defect_dir>/<subfolder>`` where
                ``subfolder`` = 'vasp_std' by default. Setting ``subfolder`` to
                ``None`` will write the ``vasp_std`` input files directly to
                the ``<defect_dir>`` folder, with no subfolders created.
            poscar (bool):
                If ``True``, writes the defect ``POSCAR`` to the generated
                folder as well. Typically not recommended, as the recommended
                workflow is to initially perform ``vasp_gam`` ground-state
                structure searching using ``ShakeNBreak``
                (https://shakenbreak.readthedocs.io) or another approach, then
                continue the ``vasp(_nkred)_std`` relaxations from the
                ground-state structures (e.g. using ``-d vasp_nkred_std`` with
                ``snb-groundstate`` (CLI) or
                ``groundstate_folder="vasp_nkred_std"`` with
                ``write_groundstate_structure`` (Python API)).
                (default: False)
            rattle (bool):
                If writing ``POSCAR``, apply random displacements to all atomic
                positions in the structure using the ``ShakeNBreak`` algorithm;
                i.e. with the displacement distances randomly drawn from a
                Gaussian distribution of standard deviation equal to 10% of the
                bulk nearest neighbour distance and using a Monte Carlo
                algorithm to penalise displacements that bring atoms closer
                than 80% of the bulk nearest neighbour distance.
                ``stdev`` and ``d_min`` can also be given as input kwargs.
                This is intended to be used as a fallback option for breaking
                symmetry to partially aid location of global minimum defect
                geometries, if ``ShakeNBreak`` structure-searching is being
                skipped. However, rattling still only finds the ground-state
                structure for <~30% of known cases of energy-lowering
                reconstructions relative to an unperturbed defect structure.
                (default: True)
            bulk (bool):
                If ``True``, the input files for a single-point calculation of
                the bulk supercell are also written to
                "{formula}_bulk/{subfolder}". (Default: False)
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
            poscar=poscar,
            rattle=rattle,
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
                poscar=True,
                rattle=False,
                vasp_xxx_attribute=self.bulk_vasp_std,
                **kwargs,
            )

    def write_nkred_std(
        self,
        defect_dir: PathLike | None = None,
        subfolder: PathLike | None = "vasp_nkred_std",
        poscar: bool = False,
        rattle: bool = True,
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
        these should be taken from the ``CONTCAR``\s of structure-searching
        calculations (e.g. via ``snb-groundstate -d vasp_nkred_std``) or, if
        not following the recommended structure-searching workflow, from the
        ``CONTCAR``\s of ``vasp_gam`` calculations. If ``POSCAR`` files are
        desired, set ``poscar=True``. If ``bulk`` is ``True``, the input files
        for a single-point calculation of the bulk supercell are also written
        to ``"{formula}_bulk/{subfolder}"``.

        Returns None and a warning if the input kpoint settings correspond to a
        Γ-only kpoint mesh (in which case ``vasp_gam`` should be used) or for
        GGA calculations (if ``LHFCALC`` is set to ``False`` in
        user_incar_settings, in which case ``vasp_std`` should be used).

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT``
        settings, and ``PotcarSet.yaml`` for the default ``POTCAR`` settings.
        **These are reasonable defaults that `roughly` match the typical values
        needed for accurate defect calculations, but usually will need to be
        modified for your specific system, such as converged ENCUT and KPOINTS,
        and NCORE / KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings
        should be consistent with those used for all defect and competing phase
        (chemical potential) calculations -- this will be automatically checked
        upon defect & competing phases parsing in ``doped``.

        The ``DefectEntry`` object is also written to a ``json.gz`` file in
        ``defect_dir`` to aid calculation provenance -- can be reloaded
        directly with ``loadfn()`` from ``monty.serialization``, or
        ``DefectEntry.from_json()``.

        Args:
            defect_dir (PathLike):
                Folder in which to create the ``VASP`` defect calculation
                inputs. Default is to use the ``DefectEntry`` name (e.g.
                ``"Y_i_C4v_O1.92_+2"`` etc.), from ``self.defect_entry.name``.
                If this attribute is not set, it is automatically generated
                according to the doped convention (using
                ``get_defect_name_from_entry()``).
            subfolder (PathLike):
                Output folder structure is ``<defect_dir>/<subfolder>`` where
                ``subfolder`` = 'vasp_nkred_std' by default. Setting
                ``subfolder`` to ``None`` will write the ``vasp_nkred_std``
                input files directly to the ``<defect_dir>`` folder, with no
                subfolders created.
            poscar (bool):
                If ``True``, writes the defect ``POSCAR`` to the generated
                folder as well. Typically not recommended, as the recommended
                workflow is to initially perform ``vasp_gam`` ground-state
                structure searching using ``ShakeNBreak``
                (https://shakenbreak.readthedocs.io) or another approach, then
                continue the ``vasp(_nkred)_std`` relaxations from the
                ground-state structures (e.g. using ``-d vasp_nkred_std`` with
                ``snb-groundstate`` (CLI) or
                ``groundstate_folder="vasp_nkred_std"`` with
                ``write_groundstate_structure`` (Python API)).
                (default: False)
            rattle (bool):
                If writing ``POSCAR``, apply random displacements to all atomic
                positions in the structure using the ``ShakeNBreak`` algorithm;
                i.e. with the displacement distances randomly drawn from a
                Gaussian distribution of standard deviation equal to 10% of the
                bulk nearest neighbour distance and using a Monte Carlo
                algorithm to penalise displacements that bring atoms closer
                than 80% of the bulk nearest neighbour distance.
                ``stdev`` and ``d_min`` can also be given as input kwargs.
                This is intended to be used as a fallback option for breaking
                symmetry to partially aid location of global minimum defect
                geometries, if ``ShakeNBreak`` structure-searching is being
                skipped. However, rattling still only finds the ground-state
                structure for <~30% of known cases of energy-lowering
                reconstructions relative to an unperturbed defect structure.
                (default: True)
            bulk (bool):
                If ``True``, the input files for a single-point calculation of
                the bulk supercell are also written to
                "{formula}_bulk/{subfolder}". (Default: False)
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
            poscar=poscar,
            rattle=rattle,
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
                poscar=True,
                rattle=False,
                vasp_xxx_attribute=self.bulk_vasp_nkred_std,
                **kwargs,
            )

    def write_ncl(
        self,
        defect_dir: PathLike | None = None,
        subfolder: PathLike | None = "vasp_ncl",
        poscar: bool = False,
        rattle: bool = True,
        bulk: bool = False,
        **kwargs,
    ):
        r"""
        Write the input files for ``VASP`` defect supercell single-point
        calculations with spin-orbit coupling (SOC) (``LSORBIT = True``)
        included, using ``vasp_ncl``.

        By default, does not generate ``POSCAR`` (input structure) files, as
        these should be taken from the ``CONTCAR``\s of ``vasp_std``
        relaxations (originally from structure-searching relaxations), or
        directly from ``ShakeNBreak`` calculations (via
        ``snb-groundstate -d vasp_ncl``) if only Γ-point reciprocal space
        sampling is required. If ``POSCAR`` files are desired, set
        ``poscar=True``.

        If ``DefectRelaxSet.soc`` is False, then this returns None and a
        warning. If the ``soc`` parameter is not set when initializing
        ``DefectRelaxSet``, then it is set to True for systems with a max
        atomic number (Z) >= 31 (i.e. further down the periodic table than Zn),
        otherwise False. If bulk is True, the input files for a single-point
        calculation of the bulk supercell are also written to
        "{formula}_bulk/{subfolder}".

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT``
        settings, and ``PotcarSet.yaml`` for the default ``POTCAR`` settings.
        **These are reasonable defaults that `roughly` match the typical values
        needed for accurate defect calculations, but usually will need to be
        modified for your specific system, such as converged ENCUT and KPOINTS,
        and NCORE / KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings
        should be consistent with those used for all defect and competing phase
        (chemical potential) calculations -- this will be automatically checked
        upon defect & competing phases parsing in ``doped``.

        The ``DefectEntry`` object is also written to a ``json.gz`` file in
        ``defect_dir`` to aid calculation provenance -- can be reloaded
        directly with ``loadfn()`` from ``monty.serialization``, or
        ``DefectEntry.from_json()``.

        Args:
            defect_dir (PathLike):
                Folder in which to create the ``VASP`` defect calculation
                inputs. Default is to use the ``DefectEntry`` name (e.g.
                ``"Y_i_C4v_O1.92_+2"`` etc.), from ``self.defect_entry.name``.
                If this attribute is not set, it is automatically generated
                according to the doped convention (using
                ``get_defect_name_from_entry()``).
            subfolder (PathLike):
                Output folder structure is ``<defect_dir>/<subfolder>`` where
                ``subfolder`` = 'vasp_ncl' by default. Setting ``subfolder`` to
                ``None`` will write the ``vasp_ncl`` input files directly to
                the ``<defect_dir>`` folder, with no subfolders created.
            poscar (bool):
                If ``True``, writes the defect ``POSCAR`` to the generated
                folder as well. Typically not recommended, as the recommended
                workflow is to initially perform ``vasp_gam`` ground-state
                structure searching (e.g. using ``ShakeNBreak``
                (https://shakenbreak.readthedocs.io)), then continue the
                ``vasp(_nkred)_std`` relaxations from the ground-state
                structures (e.g. using ``-d vasp_nkred_std`` with
                ``snb-groundstate`` (CLI) or
                ``groundstate_folder="vasp_nkred_std"`` with
                ``write_groundstate_structure`` (Python API)), first with
                ``NKRED`` if using hybrid DFT, then without, then use the
                ``vasp_std`` ``CONTCAR``\s as the input structures for final
                ``vasp_ncl`` single-point calculations.
                (default: False)
            rattle (bool):
                If writing ``POSCAR``, apply random displacements to all atomic
                positions in the structure using the ``ShakeNBreak`` algorithm;
                i.e. with the displacement distances randomly drawn from a
                Gaussian distribution of standard deviation equal to 10% of the
                bulk nearest neighbour distance and using a Monte Carlo
                algorithm to penalise displacements that bring atoms closer
                than 80% of the bulk nearest neighbour distance.
                ``stdev`` and ``d_min`` can also be given as input kwargs.
                This is intended to be used as a fallback option for breaking
                symmetry to partially aid location of global minimum defect
                geometries, if ``ShakeNBreak`` structure-searching is being
                skipped. However, rattling still only finds the ground-state
                structure for <~30% of known cases of energy-lowering
                reconstructions relative to an unperturbed defect structure.
                (default: True)
            bulk (bool):
                If ``True``, the input files for a single-point calculation of
                the bulk supercell are also written to
                "{formula}_bulk/{subfolder}". (Default: False)
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
            poscar=poscar,
            rattle=rattle,
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
                poscar=True,
                rattle=False,
                vasp_xxx_attribute=self.bulk_vasp_ncl,
                **kwargs,
            )

    def write_all(
        self,
        defect_dir: PathLike | None = None,
        poscar: bool = False,
        rattle: bool = True,
        vasp_gam: bool | None = None,
        bulk: bool | str = False,
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
            Generated if ``soc=True``. If ``soc`` is not set, then by default
            is only generated for systems with a max atomic number (Z) >= 31
            (i.e. further down the periodic table than Zn).

        If ``vasp_gam=True`` (not recommended) or ``self.vasp_std = None``
        (i.e. Γ-only `k`-point sampling converged for the kpoints settings
        used), then also outputs:

        - ``vasp_gam``:
            Γ-point only defect relaxation. Usually not needed if structure
            searching has been performed (e.g. with ``ShakeNBreak``)
            (recommended).

        By default, does not generate a ``vasp_gam`` folder unless
        ``self.vasp_std`` is ``None`` (i.e. only Γ-point sampling required for
        this system), as ``vasp_gam`` calculations should be performed with
        defect structure-searching (e.g. with ``ShakeNBreak``) and initial
        relaxations. If ``vasp_gam`` files are desired, set ``vasp_gam=True``.

        By default, ``POSCAR`` files are not generated for the
        ``vasp_(nkred_)std`` (and ``vasp_ncl`` if ``self.soc`` is True)
        folders, as these should be taken from structure-searching calculations
        (e.g. ``snb-groundstate -d vasp_nkred_std``) or, if not following the
        recommended structure-searching workflow, from the ``CONTCAR``\s of
        ``vasp_gam`` calculations. If including SOC effects (i.e.
        ``self.soc = True``), then the ``vasp_std`` ``CONTCAR``\s should be
        used as the ``vasp_ncl`` ``POSCAR``\s. If ``POSCAR`` files are desired
        for the ``vasp_(nkred_)std`` (and ``vasp_ncl``) folders, set
        ``poscar=True``.

        Input files for the single-point (static) bulk supercell reference
        calculation are also written to ``"{formula}_bulk/{subfolder}"`` if
        ``bulk`` is ``True`` (``False`` by default), where ``subfolder``
        corresponds to the final (highest accuracy) VASP calculation in the
        workflow (i.e. ``vasp_ncl`` if ``self.soc=True``, otherwise
        ``vasp_std`` or ``vasp_gam`` if only Γ-point reciprocal space sampling
        is required). If ``bulk = "all"``, then the input files for all VASP
        calculations in the workflow are written to the bulk supercell folder.

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT``
        settings, and ``PotcarSet.yaml`` for the default ``POTCAR`` settings.
        **These are reasonable defaults that `roughly` match the typical values
        needed for accurate defect calculations, but usually will need to be
        modified for your specific system, such as converged ENCUT and KPOINTS,
        and NCORE / KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings
        should be consistent with those used for all defect and competing phase
        (chemical potential) calculations -- this will be automatically checked
        upon defect & competing phases parsing in ``doped``.

        The ``DefectEntry`` object is also written to a ``json.gz`` file in
        ``defect_dir`` to aid calculation provenance -- can be reloaded
        directly with ``loadfn()`` from ``monty.serialization``, or
        ``DefectEntry.from_json()``.

        Args:
            defect_dir (PathLike):
                Folder in which to create the ``VASP`` defect calculation
                inputs. Default is to use the ``DefectEntry`` name (e.g.
                ``"Y_i_C4v_O1.92_+2"`` etc.), from ``self.defect_entry.name``.
                If this attribute is not set, it is automatically generated
                according to the doped convention (using
                ``get_defect_name_from_entry()``).
                Output folder structure is ``<defect_dir>/<subfolder>`` where
                ``subfolder`` is the name of the corresponding VASP program to
                run (e.g. ``vasp_std``).
            poscar (bool):
                If ``True``, writes the defect ``POSCAR`` to the generated
                folders as well. Typically not recommended, as the recommended
                workflow is to initially perform ``vasp_gam`` ground-state
                structure searching using ``ShakeNBreak``
                (https://shakenbreak.readthedocs.io) or another approach, then
                continue the ``vasp(_nkred)_std`` relaxations from the
                ground-state structures (e.g. using ``-d vasp_nkred_std`` with
                ``snb-groundstate`` (CLI) or
                ``groundstate_folder="vasp_nkred_std"`` with
                ``write_groundstate_structure`` (Python API)), first with
                ``NKRED`` if using hybrid DFT, then without ``NKRED``.
                (default: False)
            rattle (bool):
                If writing ``POSCAR``\s, apply random displacements to all
                atomic positions in the structures using the ``ShakeNBreak``
                algorithm; i.e. with the displacement distances randomly drawn
                from a Gaussian distribution of standard deviation equal to 10%
                of the bulk nearest neighbour distance and using a Monte Carlo
                algorithm to penalise displacements that bring atoms closer
                than 80% of the bulk nearest neighbour distance.
                ``stdev`` and ``d_min`` can also be given as input kwargs.
                This is intended to be used as a fallback option for breaking
                symmetry to partially aid location of global minimum defect
                geometries, if ``ShakeNBreak`` structure-searching is being
                skipped. However, rattling still only finds the ground-state
                structure for <~30% of known cases of energy-lowering
                reconstructions relative to an unperturbed defect structure.
                (default: True)
            vasp_gam (Optional[bool]):
                If ``True``, writes the ``vasp_gam`` input files, with defect
                ``POSCAR``. Not recommended, as the recommended workflow is to
                initially perform ``vasp_gam`` ground-state structure searching
                (e.g. using ShakeNBreak; https://shakenbreak.readthedocs.io),
                then continue the ``vasp_std`` relaxations from the
                ground-state structures. Default is ``None``, where
                ``vasp_gam`` folders are written if ``self.vasp_std`` is
                ``None`` (i.e. only Γ-point reciprocal space sampling is
                required).
            bulk (bool, str):
                If ``True``, the input files for a single-point calculation of
                the bulk supercell are also written to
                ``"{formula}_bulk/{subfolder}"``, where ``subfolder``
                corresponds to the final (highest accuracy) ``VASP``
                calculation in the workflow (i.e. ``vasp_ncl`` if
                ``self.soc=True``, otherwise ``vasp_std`` or ``vasp_gam`` if
                only Γ-point reciprocal space sampling is required). If
                ``bulk = "all"`` then the input files for all ``VASP``
                calculations in the workflow (``vasp_gam``, ``vasp_nkred_std``,
                ``vasp_std``, ``vasp_ncl`` (if applicable)) are written to the
                bulk supercell folder.
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

            if vasp_gam or (self.vasp_std is None and vasp_gam is None):
                self.write_gam(  # if vasp_std is None and vasp_gam not set to ``False``, write vasp_gam
                    defect_dir=defect_dir,
                    bulk=any("vasp_gam" in vasp_type for vasp_type in bulk_vasp),
                    poscar=poscar or vasp_gam is True,  # unperturbed poscar
                    rattle=rattle,
                    # only if `vasp_gam` explicitly set to True
                    **kwargs,
                )

            if self.vasp_std is not None:  # k-point mesh
                self.write_std(
                    defect_dir=defect_dir,
                    poscar=poscar,
                    rattle=rattle,
                    bulk=any("vasp_std" in vasp_type for vasp_type in bulk_vasp),
                    **kwargs,
                )

            if self.vasp_nkred_std is not None:  # k-point mesh and not GGA
                self.write_nkred_std(
                    defect_dir=defect_dir,
                    poscar=poscar,
                    rattle=rattle,
                    bulk=any("vasp_nkred_std" in vasp_type for vasp_type in bulk_vasp),
                    **kwargs,
                )

        if self.soc:  # SOC
            self.write_ncl(
                defect_dir=defect_dir,
                poscar=poscar,
                rattle=rattle,
                bulk=any("vasp_ncl" in vasp_type for vasp_type in bulk_vasp),
                **kwargs,
            )

    def __repr__(self):
        """
        Returns a string representation of the ``DefectRelaxSet`` object.
        """
        formula = self.bulk_supercell.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
        properties, methods = _doped_obj_properties_methods(self)
        return (
            f"doped DefectRelaxSet for bulk composition {formula}, and defect entry "
            f"{self.defect_entry.name}. Available attributes:\n{properties}\n\n"
            f"Available methods:\n{methods}"
        )


class DefectsSet(MSONable):
    """
    Class for generating input files for ``VASP`` defect calculations for a set
    of ``doped``/``pymatgen`` ``DefectEntry`` objects.
    """

    def __init__(
        self,
        defect_entries: DefectsGenerator | dict[str, DefectEntry] | list[DefectEntry] | DefectEntry,
        soc: bool | None = None,
        user_incar_settings: dict | None = None,
        user_kpoints_settings: dict | Kpoints | None = None,
        user_potcar_functional: str = "PBE",
        user_potcar_settings: dict | None = None,
        **kwargs,  # to allow POTCAR testing on GH Actions
    ):
        r"""
        Creates a dictionary of: ``{defect name: DefectRelaxSet}``.

        ``DefectRelaxSet`` has the attributes:

        - ``DefectRelaxSet.vasp_gam``:
            ``DefectDictSet`` for Gamma-point only relaxation. Usually not
            needed if structure searching (e.g. ``ShakeNBreak``) has been
            performed (recommended), unless only Γ-point `k`-point sampling is
            required (converged) for your system, and no ``vasp_std``
            calculations with multiple `k`-points are required (determined from
            kpoint settings).
        - ``DefectRelaxSet.vasp_nkred_std``:
            ``DefectDictSet`` for relaxation with a kpoint mesh and using
            ``NKRED``. Not generated for GGA calculations (if ``LHFCALC`` is
            set to ``False`` in ``user_incar_settings``) or if only Gamma
            `k`-point sampling is required.
        - ``DefectRelaxSet.vasp_std``:
            ``DefectDictSet`` for relaxation with a kpoint mesh, not using
            ``NKRED``. Not generated if only Gamma kpoint sampling is required.
        - ``DefectRelaxSet.vasp_ncl``:
            ``DefectDictSet`` for single-point (static) energy calculation with
            SOC included. Generated if ``soc=True``. If ``soc`` is not set,
            then by default is only generated for systems with a max atomic
            number (Z) >= 31 (i.e. further down the periodic table than Zn).

        where ``DefectDictSet`` is an extension of ``pymatgen``'s
        ``VaspInputSet`` class for defect calculations, with ``incar``,
        ``poscar``, ``kpoints`` and ``potcar`` attributes for the corresponding
        VASP defect calculations (see docstring). Also creates the
        corresponding ``bulk_vasp_...`` attributes for single-point (static)
        energy calculations of the bulk (pristine, defect-free) supercell. This
        needs to be calculated once with the same settings as the final defect
        calculations, for the later calculation of defect formation energies.

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` settings, and
        ``PotcarSet.yaml`` for the default ``POTCAR`` settings.

        Note that any changes to the default ``INCAR``/``POTCAR`` settings
        should be consistent with those used for all defect and competing phase
        (chemical potential) calculations -- this will be automatically checked
        upon defect & competing phases parsing in ``doped``.

        Args:
            defect_entries (``DefectsGenerator``, dict/list of ``DefectEntry``\s, or ``DefectEntry``):
                Either a ``DefectsGenerator`` object, or a dictionary/list of
                ``DefectEntry``\s, or a single ``DefectEntry`` object, for
                which to generate VASP input files.
                If a ``DefectsGenerator`` object or a dictionary (->
                ``{defect name: DefectEntry}``), the defect folder names
                will be set equal to ``defect name``. If a list or single
                ``DefectEntry`` object is provided, the defect folder names
                will be set equal to ``DefectEntry.name`` if the ``name``
                attribute is set, otherwise generated according to the
                ``doped`` convention (see ``doped.generation``).
                Defect charge states are taken from
                ``DefectEntry.charge_state``.
            soc (bool):
                Whether to generate ``vasp_ncl`` ``DefectDictSet`` attribute
                for spin-orbit coupling single-point (static) energy
                calculations. If not set, then by default is set to ``True`` if
                the max atomic number (Z) in the structure is >= 31 (i.e.
                further down the periodic table than Zn).
            user_incar_settings (dict):
                Dictionary of user ``INCAR`` settings (AEXX, NCORE etc.) to
                override default settings. Highly recommended to look at output
                ``INCAR``\s or the ``RelaxSet.yaml`` and ``DefectSet.yaml``
                files in the ``doped/VASP_sets`` folder, to see what the
                default ``INCAR`` settings are. Note that any flags that aren't
                numbers or ``True/False`` need to be input as strings with
                quotation marks (e.g. ``{"ALGO": "All"}``).
                (default: None)
            user_kpoints_settings (dict or Kpoints):
                Dictionary of user ``KPOINTS`` settings (in ``pymatgen``
                ``VaspInputSet`` format) e.g., ``{"reciprocal_density": 123}``,
                or a ``Kpoints`` object, to use for the ``vasp_std``,
                ``vasp_nkred_std`` and ``vasp_ncl`` ``DefectDictSet``\s (Γ-only
                for ``vasp_gam``). Default is Gamma-centred,
                ``reciprocal_density = 100`` [Å⁻³].
            user_potcar_functional (str):
                ``POTCAR`` functional to use. Default is "PBE" and if this
                fails, tries "PBE_52", then "PBE_54".
            user_potcar_settings (dict):
                Override the default ``POTCAR``\s, e.g. ``{"Li": "Li_sv"}``.
                See ``doped/VASP_sets/PotcarSet.yaml`` for the default
                ``POTCAR`` set.
            **kwargs: Additional kwargs to pass to each ``DefectRelaxSet()``.

        Key Attributes:
            defect_sets (Dict):
                Dictionary of ``{defect name: DefectRelaxSet}``.
            defect_entries (Dict):
                Dictionary of ``{defect name: DefectEntry}`` for the input
                defect species, for which to generate ``VASP`` input files.
            bulk_vasp_gam (DefectDictSet):
                ``DefectDictSet`` for a `bulk` Γ-point-only single-point
                (static) supercell calculation. Often not used, as the bulk
                supercell only needs to be calculated once with the same
                settings as the final defect calculations, which may be with
                ``vasp_std`` or ``vasp_ncl``.
            bulk_vasp_nkred_std (DefectDictSet):
                ``DefectDictSet`` for a single-point (static) `bulk`
                ``vasp_std`` supercell calculation (i.e. with a non-Γ-only
                kpoint mesh) and ``NKRED(X,Y,Z)`` ``INCAR`` tag(s) to
                downsample kpoints for the HF exchange part of the hybrid DFT
                calculation. Not generated for GGA calculations (if ``LHFCALC``
                is set to ``False`` in ``user_incar_settings``) or if only
                Gamma kpoint sampling is required.
            bulk_vasp_std (DefectDictSet):
                ``DefectDictSet`` for a single-point (static) `bulk`
                ``vasp_std`` supercell calculation with a non-Γ-only kpoint
                mesh, not using ``NKRED``. Not generated if only Gamma kpoint
                sampling is required.
            bulk_vasp_ncl (DefectDictSet):
                ``DefectDictSet`` for single-point (static) energy calculation
                of the `bulk` supercell with SOC included. Generated if
                ``soc=True``. If ``soc`` is not set, then by default is only
                generated for systems with a max atomic number (Z) >= 31 (i.e.
                further down the periodic table than Zn).
            bulk_supercell (Structure):
                Supercell structure of the bulk (pristine) material.
            json_obj (Union[Dict, DefectsGenerator]):
                Either the ``DefectsGenerator`` object if input
                ``defect_entries`` is a ``DefectsGenerator`` object, otherwise
                the ``defect_entries`` dictionary, which will be written to
                file when ``write_files()`` is called, to aid calculation
                provenance.
            json_name (PathLike):
                Name of the ``JSON`` file to save the ``json_obj`` to.

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

        self.defect_sets: dict[str, DefectRelaxSet] = {}

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
        self.bulk_supercell = defect_relax_set.bulk_supercell
        self.bulk_vasp_gam = defect_relax_set.bulk_vasp_gam
        with warnings.catch_warnings():  # ignore vasp_std kpoints warnings if vasp_gam only here
            warnings.filterwarnings("ignore", "With the current")
            self.bulk_vasp_nkred_std = defect_relax_set.bulk_vasp_nkred_std
            self.bulk_vasp_std = defect_relax_set.bulk_vasp_std
            if self.bulk_vasp_std is None:  # print one info message about this
                defect_relax_set._check_vstd_kpoints(defect_relax_set._vasp_std, warn=False, info=True)

        self.bulk_vasp_ncl = defect_relax_set.bulk_vasp_ncl

    def _format_defect_entries_input(
        self,
        defect_entries: DefectsGenerator | dict[str, DefectEntry] | list[DefectEntry] | DefectEntry,
    ) -> tuple[dict[str, DefectEntry], str, dict[str, DefectEntry] | DefectsGenerator]:
        r"""
        Helper function to format input ``defect_entries`` into a named
        dictionary of ``DefectEntry`` objects.

        Also returns the name of the JSON file and object to serialise when
        writing the VASP input to files. This is the DefectsGenerator object if
        ``defect_entries`` is a ``DefectsGenerator`` object, otherwise the
        dictionary of ``DefectEntry`` objects.

        Args:
            defect_entries (``DefectsGenerator``, dict/list of ``DefectEntry``\s, or ``DefectEntry``):
                Either a ``DefectsGenerator`` object, or a dictionary/list of
                ``DefectEntry``\s, or a single ``DefectEntry`` object, for
                which to generate ``VASP`` input files.
                If a ``DefectsGenerator`` object or a dictionary (->
                ``{defect name: DefectEntry}``), the defect folder names will
                be set equal to ``defect name``. If a list or single
                ``DefectEntry`` object is provided, the defect folder names
                will be set equal to ``DefectEntry.name`` if the ``name``
                attribute is set, otherwise generated according to the
                ``doped`` convention (see ``doped.generation``).
        """
        json_filename = "defect_entries.json.gz"  # global statement in case, but should be skipped
        json_obj = defect_entries
        if type(defect_entries).__name__ == "DefectsGenerator":
            defect_entries = cast(DefectsGenerator, defect_entries)
            formula = defect_entries.primitive_structure.composition.get_reduced_formula_and_factor(
                iupac_ordering=True
            )[0]
            json_filename = f"{formula}_defects_generator.json.gz"
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
            json_filename = f"{formula}_defect_entries.json.gz"
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
        defect_species, defect_relax_set, output_path, poscar, rattle, vasp_gam, bulk, kwargs = args
        defect_dir = os.path.join(output_path, defect_species)
        defect_relax_set.write_all(
            defect_dir=defect_dir,
            poscar=poscar,
            rattle=rattle,
            vasp_gam=vasp_gam,
            bulk=bulk,
            **kwargs,
        )

    def write_files(
        self,
        output_path: PathLike = ".",
        poscar: bool = False,
        rattle: bool = True,
        vasp_gam: bool | None = None,
        bulk: bool | str = True,
        processes: int | None = None,
        **kwargs,
    ):
        r"""
        Write VASP input files to folders for all defects in
        ``self.defect_entries``. Folder names are set to the key of the
        DefectRelaxSet in ``self.defect_sets`` (same as self.defect_entries
        keys, see ``DefectsSet`` docstring).

        For each defect folder, the following subfolders are generated:

        - ``vasp_nkred_std``:
            Defect relaxation with a kpoint mesh and using ``NKRED``. Not
            generated for GGA calculations (if ``LHFCALC`` is set to ``False``
            in ``user_incar_settings``) or if only Γ-point sampling required.
        - ``vasp_std``:
            Defect relaxation with a kpoint mesh, not using ``NKRED``. Not
            generated if only Γ-point sampling required.
        - ``vasp_ncl``:
            Singlepoint (static) energy calculation with SOC included.
            Generated if ``soc=True``. If ``soc`` is not set, then by default
            is only generated for systems with a max atomic number (Z) >= 31
            (i.e. further down the periodic table than Zn).

        If ``vasp_gam=True`` (not recommended) or ``self.vasp_std = None``
        (i.e. Γ-only `k`-point sampling converged for the kpoints settings
        used), then also outputs:

        - ``vasp_gam``:
            Γ-point only defect relaxation. Usually not needed if structure
            searching has been performed (e.g. with ``ShakeNBreak``)
            (recommended).

        By default, does not generate a ``vasp_gam`` folder unless
        ``DefectRelaxSet.vasp_std`` is ``None`` (i.e. only Γ-point sampling
        required for this system), as ``vasp_gam`` calculations should be
        performed with defect structure-searching (e.g. with ``ShakeNBreak``)
        and initial relaxations. If ``vasp_gam`` files are desired, set
        ``vasp_gam=True``.

        By default, ``POSCAR`` files are not generated for the
        ``vasp_(nkred_)std`` (and ``vasp_ncl`` if ``self.soc`` is ``True``)
        folders, as these should be taken from structure-searching calculations
        (e.g. ``snb-groundstate -d vasp_nkred_std``) or, if not following the
        recommended structure-searching workflow, from the ``CONTCAR``\s of
        ``vasp_gam`` calculations. If including SOC effects (i.e.
        ``self.soc = True``), then the ``vasp_std`` ``CONTCAR``\s should be
        used as the ``vasp_ncl`` ``POSCAR``\s. If ``POSCAR`` files are desired
        for the ``vasp_(nkred_)std`` (and ``vasp_ncl``) folders, set
        ``poscar=True``.

        Input files for the single-point (static) bulk supercell reference
        calculation are also written to ``"{formula}_bulk/{subfolder}"`` if
        ``bulk`` is ``True`` (default), where ``subfolder`` corresponds to the
        final (highest accuracy) VASP calculation in the workflow (i.e.
        ``vasp_ncl`` if ``self.soc=True``, otherwise ``vasp_std`` or
        ``vasp_gam`` if only Γ-point reciprocal space sampling is required). If
        ``bulk = "all"``, then the input files for all VASP calculations
        (gam/std/ncl) are written to the bulk supercell folder, or if
        ``bulk = False``, then no bulk folder is created.

        The ``DefectEntry`` objects are also written to ``json.gz`` files in
        the defect folders, as well as ``self.defect_entries``
        (``self.json_obj``) in the top folder, to aid calculation provenance
        -- these can be reloaded directly with ``loadfn()`` from
        ``monty.serialization``, or individually with
        ``DefectEntry.from_json()``.

        See the ``RelaxSet.yaml`` and ``DefectSet.yaml`` files in the
        ``doped/VASP_sets`` folder for the default ``INCAR`` and ``KPOINT``
        settings, and ``PotcarSet.yaml`` for the default ``POTCAR`` settings.
        **These are reasonable defaults that `roughly` match the typical values
        needed for accurate defect calculations, but usually will need to be
        modified for your specific system, such as converged ENCUT and KPOINTS,
        and NCORE / KPAR matching your HPC setup.**

        Note that any changes to the default ``INCAR``/``POTCAR`` settings
        should be consistent with those used for all defect and competing phase
        (chemical potential) calculations -- this will be automatically checked
        upon defect & competing phases parsing in ``doped``.

        Args:
            output_path (PathLike):
                Folder in which to create the VASP defect calculation folders.
                Default is the current directory ("."). Output folder structure
                is ``<output_path>/<defect name>/<subfolder>`` where
                ``defect name`` is the key of the DefectRelaxSet in
                ``self.defect_sets`` (same as ``self.defect_entries`` keys, see
                ``DefectsSet`` docstring) and ``subfolder`` is the name of the
                corresponding VASP program to run (e.g. ``vasp_std``).
            poscar (bool):
                If ``True``, writes the defect ``POSCAR`` to the generated
                folders as well. Typically not recommended, as the recommended
                workflow is to initially perform ``vasp_gam`` ground-state
                structure searching using ``ShakeNBreak``
                (https://shakenbreak.readthedocs.io) or another approach, then
                continue the ``vasp(_nkred)_std`` relaxations from the
                ground-state structures (e.g. using ``-d vasp_nkred_std`` with
                ``snb-groundstate`` (CLI) or
                ``groundstate_folder="vasp_nkred_std"`` with
                ``write_groundstate_structure`` (Python API)), first with
                ``NKRED`` if using hybrid DFT, then without ``NKRED``.
                (default: False)
            rattle (bool):
                If writing ``POSCAR``\s, apply random displacements to all
                atomic positions in the structures using the ``ShakeNBreak``
                algorithm; i.e. with the displacement distances randomly drawn
                from a Gaussian distribution of standard deviation equal to 10%
                of the bulk nearest neighbour distance and using a Monte Carlo
                algorithm to penalise displacements that bring atoms closer
                than 80% of the bulk nearest neighbour distance.
                ``stdev`` and ``d_min`` can also be given as input kwargs.
                This is intended to be used as a fallback option for breaking
                symmetry to partially aid location of global minimum defect
                geometries, if ``ShakeNBreak`` structure-searching is being
                skipped. However, rattling still only finds the ground-state
                structure for <~30% of known cases of energy-lowering
                reconstructions relative to an unperturbed defect structure.
                (default: True)
            vasp_gam (Optional[bool]):
                If ``True``, writes the ``vasp_gam`` input files, with defect
                ``POSCAR``\s. Not recommended, as the recommended workflow is
                to initially perform ``vasp_gam`` ground-state structure
                searching (e.g. using ``ShakeNBreak``;
                https://shakenbreak.readthedocs.io), then continue the
                ``vasp_std`` relaxations from the ground-state structures.
                Default is ``None``, where ``vasp_gam`` folders are written if
                ``self.vasp_std`` is ``None`` (i.e. only Γ-point reciprocal
                space sampling is required).
            bulk (bool, str):
                If ``True``, the input files for a single-point calculation of
                the bulk supercell are also written to
                ``"{formula}_bulk/{subfolder}"``, where ``subfolder``
                corresponds to the final (highest accuracy) ``VASP``
                calculation in the workflow (i.e. ``vasp_ncl`` if
                ``self.soc=True``, otherwise ``vasp_std`` or ``vasp_gam`` if
                only Γ-point reciprocal space sampling is required). If
                ``bulk = "all"`` then the input files for all ``VASP``
                calculations in the workflow (``vasp_gam``, ``vasp_nkred_std``,
                ``vasp_std``, ``vasp_ncl`` (if applicable)) are written to the
                bulk supercell folder.
                (Default: False)
            processes (int):
                Number of processes to use for ``multiprocessing`` for file
                writing. If ``None`` (default), then is dynamically set to the
                optimal value for the number of folders to write.
                (Default: None)
            **kwargs:
                Keyword arguments to pass to ``DefectDictSet.write_input()``.
        """
        # TODO: If POTCARs not setup, warn and only write neutral defect folders, with INCAR, KPOINTS and
        #  (if poscar) POSCAR? And bulk

        args_list = [
            (
                defect_species,
                defect_relax_set,
                output_path,
                poscar,
                rattle,
                vasp_gam,
                bulk if i == len(self.defect_sets) - 1 else False,  # write bulk folder(s) for last defect
                kwargs,
            )
            for i, (defect_species, defect_relax_set) in enumerate(self.defect_sets.items())
        ]
        if processes is None:  # best setting for number of processes, from testing
            mp = get_mp_context()
            processes = min(round(len(args_list) / 30), mp.cpu_count() - 1)

        if processes > 1:
            with pool_manager(processes) as pool:
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
        Returns a string representation of the ``DefectsSet`` object.
        """
        formula = next(
            iter(self.defect_entries.values())
        ).defect.structure.composition.get_reduced_formula_and_factor(iupac_ordering=True)[0]
        properties, methods = _doped_obj_properties_methods(self)
        return (
            f"doped DefectsSet for bulk composition {formula}, with {len(self.defect_entries)} "
            f"defect entries in self.defect_entries. Available attributes:\n{properties}\n\n"
            f"Available methods:\n{methods}"
        )

    def __getattr__(self, attr):
        """
        Redirects an unknown attribute/method call to the ``defect_sets``
        dictionary attribute, if the attribute doesn't exist in ``DefectsSet``.
        """
        # Return the attribute if it exists in self.__dict__
        if attr in self.__dict__:
            return self.__dict__[attr]

        # Check if the attribute exists in defect_sets:
        if hasattr(self.defect_sets, attr):
            return getattr(self.defect_sets, attr)

        # If all else fails, raise an AttributeError
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __getitem__(self, key):
        """
        Makes ``DefectsSet`` object subscriptable, so that it can be indexed
        like a dictionary, using the ``defect_sets`` dictionary attribute.
        """
        return self.defect_sets[key]

    def __setitem__(self, key, value):
        """
        Set the value of a specific key (defect name) in the ``defect_sets``
        dictionary.

        Note that other ``DefectsSet`` attributes, like ``self.soc`` (whether
        to perform SOC calculations) and ``self.bulk_vasp...`` are not changed.
        """
        # check the input, must be a ``DefectRelaxSet`` object:
        if not isinstance(value, DefectRelaxSet):
            raise TypeError(f"Value must be a DefectRelaxSet object, not {type(value).__name__}")

        # check it has the same bulk supercell, and warn if not:
        if value.bulk_supercell != self.bulk_supercell:
            warnings.warn(
                "Note that the bulk supercell of the input DefectRelaxSet differs from that of the "
                "DefectsSet. This could lead to inaccuracies in parsing/predictions if the bulk "
                "supercells are not the same!\n"
                f"DefectRelaxSet bulk supercell:\n{value.bulk_supercell}\n"
                f"DefectsSet bulk supercell:\n{self.bulk_supercell}"
            )

        self.defect_sets[key] = value

    def __delitem__(self, key):
        """
        Deletes the specified ``DefectRelaxSet`` from the ``defect_sets``
        dictionary.
        """
        del self.defect_sets[key]

    def __contains__(self, key):
        """
        Returns True if the ``defect_sets`` dictionary contains the specified
        defect name.
        """
        return key in self.defect_sets

    def __len__(self):
        r"""
        Returns the number of ``DefectRelaxSet``\s in the ``defect_sets``
        dictionary.
        """
        return len(self.defect_sets)

    def __iter__(self):
        """
        Returns an iterator over the ``defect_sets`` dictionary.
        """
        return iter(self.defect_sets)


# TODO: Go through and update docstrings with descriptions all the default behaviour (INCAR,
#  KPOINTS settings etc)
# TODO: Have optional parameter to output DefectRelaxSet jsons to written folders as well (but off by
#  default)?
# TODO: Likewise, add same to/from json etc. functions for DefectRelaxSet. __Dict__ methods apply
#  to `.defect_sets` etc?
# TODO: Implement renaming folders like SnB if we try to write a folder that already exists,
#  and the structures don't match (otherwise overwrite)
