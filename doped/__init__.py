"""
`doped` is a python package for managing solid-state defect calculations, with
functionality to generate defect structures and relevant competing phases (for
chemical potentials), interface with ShakeNBreak
(https://shakenbreak.readthedocs.io) for defect structure-searching (see
https://www.nature.com/articles/s41524-023-00973-1), write VASP input files for
defect supercell calculations, and automatically parse and analyse the results.
"""
import warnings
from importlib.metadata import PackageNotFoundError, version

from packaging.version import parse
from pymatgen.io.vasp.inputs import UnknownPotcarWarning
from pymatgen.io.vasp.sets import BadInputSetWarning

# if date.today().weekday() in [5, 6]:
#     print("""Working on the weekend, like usual...\n""")
# if date.today().weekday() == 5:
#     print("Seriously though, everyone knows Saturday's for the boys/girls...\n")
# Killed by multiprocessing, # never forget


def _check_pmg_compatibility():
    try:
        v_pmg_def = version("pymatgen-analysis-defects")
    except PackageNotFoundError:
        v_pmg_def = False

    try:
        v_pmg = version("pymatgen")
    except Exception:
        v_pmg = False

    if not v_pmg_def:
        raise TypeError(
            "You do not have the `pymatgen-analysis-defects` package installed, which is required by "
            "`doped`. Please install `pymatgen-analysis-defects` (with `pip install "
            "pymatgen-analysis-defects`) and restart the kernel."
        )

    if parse(v_pmg) < parse("2022.7.25"):
        raise TypeError(
            f"You have the version {v_pmg} of `pymatgen` installed, which is incompatible with `doped`. "
            f"Please update (with `pip install --upgrade pymatgen`) and restart the kernel."
        )


_check_pmg_compatibility()


def _ignore_pmg_warnings():
    # globally ignore these POTCAR warnings
    warnings.filterwarnings("ignore", category=UnknownPotcarWarning)
    warnings.filterwarnings("ignore", category=BadInputSetWarning)
    warnings.filterwarnings("ignore", message="No POTCAR file with matching TITEL fields")
    warnings.filterwarnings("ignore", message="Ignoring unknown variable type")
    warnings.filterwarnings(
        "ignore", message="POTCAR data with symbol"
    )  # Ignore POTCAR warnings because Pymatgen incorrectly detecting POTCAR types

    # Ignore because comment after 'ALGO = Normal' causes this unnecessary warning:
    warnings.filterwarnings("ignore", message="Hybrid functionals only support")

    warnings.filterwarnings("ignore", message="Use get_magnetic_symmetry()")
    warnings.filterwarnings("ignore", message="Use of properties is now deprecated")


_ignore_pmg_warnings()
