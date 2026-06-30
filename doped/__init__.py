"""
``doped`` is a python package for managing solid-state defect calculations,
with functionality to generate defect structures and relevant competing phases
(for chemical potentials), interface with |ShakeNBreak|
(https://shakenbreak.readthedocs.io) for defect structure-searching (see
https://www.nature.com/articles/s41524-023-00973-1), write VASP input files for
defect supercell calculations, and automatically parse and analyse the results.
"""

from importlib.metadata import PackageNotFoundError, version

# set __version__ for older users who use this convention:
try:
    __version__ = version("doped")  # from package metadata (pyproject.toml)
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback for local development or if package isn't installed
