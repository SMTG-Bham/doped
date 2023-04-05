Change Log
==========

v23.4.5
----------
- Start keeping a `CHANGELOG`
- `README` updates to give step-by-step instructions on setting up MP API key, `POTCAR`s for `pymatgen` and virtual `conda` environments for `doped` and `ShakeNBreak`
- Major overhaul of `vasp_input` functions setup to be far more streamlined and customisable.
- Major overhaul of `chemical_potentials` code; now with improved algorithm for selecting potential competing phases
- Update of example notebooks
- Add tests for parsing calculations, chemical_potentials and vasp_input
- Add GH Actions workflows (for tests, GH releases and pypi packaging)
- Adopt recommended versioning convention based on dates
- General tidy up, docstring padding, formatting and `TODO` addition/removal
- Ensure all inputs/outputs are `JSON`able, now recommending this for better forward/backward compatibility
- Refactor `dope_stuff` to `plotting` and `analysis` to be more clear and PROfessional, yo
- Refactor from hard-coded defaults / slightly-less-human-readable `json` files to `yaml` files with default settings.
- Refactor `defectsmaker` output, more efficient, cleaner and informative


