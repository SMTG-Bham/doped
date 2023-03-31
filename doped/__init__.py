import warnings
from datetime import date
from importlib.metadata import PackageNotFoundError, version

from pymatgen.io.vasp.inputs import UnknownPotcarWarning

if date.today().weekday() in [5, 6]:
    print("""Working on the weekend, like usual...\n""")
if date.today().weekday() == 5:
    print("Seriously though, everyone knows Saturday's for the boys/girls...\n")


def _check_pmg_compatibility():
    try:
        v_pmg_def = version("pymatgen-analysis-defects")
    except PackageNotFoundError:
        v_pmg_def = False

    try:
        v_pmg = version("pymatgen")
    except Exception:
        v_pmg = False

    if v_pmg_def:
        raise TypeError(
            "You have the `pymatgen-analysis-defects` package installed, which is currently "
            "incompatible with `doped`. Please uninstall `pymatgen-analysis-defects` (with `pip "
            "uninstall pymatgen-analysis-defects`) and restart the kernel."
        )

    if v_pmg > "2022.7.25":
        raise TypeError(
            f"You have the version {v_pmg} of `pymatgen` installed, which is currently "
            f"incompatible with `doped`. Please revert this package (with `pip install "
            f"pymatgen==2022.7.25`) and restart the kernel."
        )


_check_pmg_compatibility()

# globally ignore these POTCAR warnings
warnings.filterwarnings("ignore", category=UnknownPotcarWarning)
warnings.filterwarnings("ignore", message="No POTCAR file with matching TITEL fields")
warnings.filterwarnings("ignore", message="Ignoring unknown variable type")
warnings.filterwarnings(
        "ignore", message="POTCAR data with symbol"
    )  # Ignore POTCAR warnings because Pymatgen incorrectly detecting POTCAR types
# Ignore because comment after 'ALGO = Normal' causes this unnecessary warning:
warnings.filterwarnings("ignore", message="Hybrid functionals only support")

# until updated from pymatgen==2022.7.25 :
warnings.filterwarnings(
    "ignore", message="Using `tqdm.autonotebook.tqdm` in notebook mode"
)
warnings.filterwarnings("ignore", message="`np.int` is a deprecated alias for the builtin `int`")
warnings.filterwarnings("ignore", message="Use get_magnetic_symmetry()")
