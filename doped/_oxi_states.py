"""
Functions for guessing oxidation states which are run in separate processes (to
allow timeouts; see ``guess_and_set_oxi_states_with_timeout`` in
``doped.core``).

These are kept in this separate module, which should only import the minimal
required packages, as each of these processes must first import the module of
its target function -- so the imports here set their start-up times (~0.1 s,
but much longer on Windows or slow filesystems).
"""

import contextlib

import numpy as np
from pymatgen.core.structure import Structure


def _guess_and_set_struct_oxi_states_icsd_prob(
    structure: Structure, try_without_max_sites: bool = False
) -> Structure | bool:
    """
    Tries to guess (and set) the oxidation states of the input structure, using
    the ``pymatgen``-tabulated ICSD oxidation state probabilities.

    Args:
        structure (|Structure|):
            The structure for which to guess the oxidation states.
        try_without_max_sites (bool):
            Whether to try to guess the oxidation states
            without using the ``max_sites=-1`` argument (``True``)(which
            attempts to use the reduced composition for guessing oxi states) or
            not (``False``; default).

    Returns:
        Structure:
            The structure with oxidation states guessed and set, or ``False``
            if oxidation states could not be guessed.
    """
    structure = structure.copy()  # don't modify original structure
    if try_without_max_sites:
        with contextlib.suppress(Exception):
            structure.add_oxidation_state_by_guess()
            # check all oxidation states are whole numbers:
            if all(np.isclose(int(specie.oxi_state), specie.oxi_state) for specie in structure.species):
                return structure

    # else try to use the reduced cell since oxidation state assignment scales poorly with system size:
    try:  # retry with broader guesses if oxi states are all zero or not whole numbers:
        for kwargs in ({"max_sites": -1}, {"max_sites": -1, "all_oxi_states": True}, {}):
            structure.add_oxidation_state_by_guess(**kwargs)
            if not all(specie.oxi_state == 0 for specie in structure.species) and all(
                np.isclose(int(specie.oxi_state), specie.oxi_state) for specie in structure.species
            ):
                break
        if not all(np.isclose(int(specie.oxi_state), specie.oxi_state) for specie in structure.species):
            return False  # non-integer (e.g. mixed-valence average) oxidation states from every guess
    except Exception:
        structure.add_oxidation_state_by_guess()

    if all(hasattr(site.specie, "oxi_state") for site in structure.sites) and all(
        isinstance(site.specie.oxi_state, int | float) for site in structure.sites
    ):
        return structure

    return False


def _guess_and_set_struct_oxi_states_icsd_prob_process(
    structure, queue, try_without_max_sites=False, started=None
):
    """
    Implements the ``_guess_and_set_struct_oxi_states_icsd_prob`` function
    above, but also putting the results into the supplied ``multiprocessing``
    queue object (for use with timeouts via ``Process``), after first setting
    the ``started`` ``multiprocessing`` event (if given) to signal that the
    process is up and running.

    For internal ``doped`` usage.
    """
    if started is not None:
        started.set()
    try:
        structure_with_oxi = _guess_and_set_struct_oxi_states_icsd_prob(structure, try_without_max_sites)
    except Exception:  # e.g. unknown elements; must put a result, else ``queue.get()`` hangs
        structure_with_oxi = False
    queue.put(structure_with_oxi or False)
