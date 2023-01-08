from datetime import date
from importlib.metadata import version, PackageNotFoundError

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
            f"You have the `pymatgen-analysis-defects` package installed, "
            f"which is currently incompatible with `doped`. "
            f"Please uninstall `pymatgen-analysis-defects` (with pip uninstall pymatgen-analysis-defects) "
            f"and try again."
        )

    if v_pmg > "2022.7.25":
        raise TypeError(
            f"You have the version {v_pmg} of `pymatgen` installed, which is currently "
            f"incompatible with doped. "
            f"Please revert this package (with `pip install pymatgen==2022.7.25`) and restart the kernel."
        )
_check_pmg_compatibility()

