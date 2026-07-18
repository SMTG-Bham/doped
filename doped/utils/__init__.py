"""
Submodule for utility functions in doped.

``doped.utils.__init__`` contains utility functions and context managers for
handling of warnings and multi-processing.
"""

import contextlib
import importlib
import inspect
import logging
import multiprocessing
import os
import warnings
from functools import cache
from typing import Any

from pymatgen.io.vasp.inputs import UnknownPotcarWarning
from pymatgen.io.vasp.sets import BadInputSetWarning


@contextlib.contextmanager
def suppress_logging(level=logging.CRITICAL):
    """
    Context manager to catch and suppress logging messages.
    """
    previous_level = logging.root.manager.disable  # store the current logging level
    logging.disable(level)  # disable logging at the specified level
    try:
        yield
    finally:
        logging.disable(previous_level)  # restore the original logging level


@contextlib.contextmanager
def patch_vise_for_windows():
    """
    Context manager to patch
    ``vise.defaults.UserSettings._make_yaml_file_list``, so that it returns an
    empty list.

    Fixes an issue where this function gives an infinite recursive search on
    Windows, causing hanging.
    """
    try:
        vd = importlib.import_module("vise.defaults")
        orig = vd.UserSettings._make_yaml_file_list
        vd.UserSettings._make_yaml_file_list = lambda *args, **kwargs: []
        yield
    finally:  # restore original
        vd.UserSettings._make_yaml_file_list = orig


@contextlib.contextmanager
def vise_handling(level=logging.CRITICAL):
    r"""
    Tame ``vise``/``pydefect`` side effects, by combining
    :func:`suppress_logging`, ``warnings.catch_warnings()`` and
    :func:`patch_vise_for_windows`.

    The steps are ordered to handle two things that must happen _before_
    ``vise.defaults`` is first imported anywhere:

    1. ``vise.util.logger.get_logger`` is replaced with ``logging.getLogger``,
       to avoid repeated ``vise`` INFO messages (and duplicate handlers) under
       parallelism. Only ``vise.util.logger`` is imported for this, which does
       _not_ pull in ``vise.defaults``, so the patch is in effect before
       ``vise.defaults`` builds its module-level logger.

    2. Import ``vise.defaults`` now, within ``catch_warnings()`` so its
       ``warnings.simplefilter("ignore", UserWarning)`` is reverted on exit.
       Otherwise this fires the first time ``vise.defaults`` is imported --
       which, with multiprocessing parsing, happens lazily during
       *result unpickling* with ``Pool`` (importing a ``pydefect``
       ``BandEdgeStates`` object -> ``pydefect.defaults`` ->
       ``vise.defaults``), i.e. _outside_ any ``vise_handling()`` block.

    Calling this once at ``doped`` import (below) leaves ``vise.defaults`` in
    ``sys.modules`` with both fixes applied, so later imports are no-ops and
    never re-trigger warning suppression.
    """
    with suppress_logging(level), warnings.catch_warnings():
        import vise.util.logger

        vise.util.logger.get_logger = logging.getLogger

        with patch_vise_for_windows():  # imports ``vise.defaults`` / builds the ``Defaults`` singleton
            yield


with vise_handling():  # tame ``vise``/``pydefect`` side effects at import time (see :func:`vise_handling`)
    pass


def _ignore_pmg_warnings():
    # globally ignore these POTCAR warnings; `message` only needs to match start of message
    warnings.filterwarnings("ignore", category=UnknownPotcarWarning)
    warnings.filterwarnings("ignore", category=BadInputSetWarning)
    warnings.filterwarnings("ignore", message="No POTCAR file with matching TITEL fields")
    warnings.filterwarnings("ignore", message="Ignoring unknown variable type")

    # Ignore because comment after 'ALGO = Normal' causes this unnecessary warning:
    warnings.filterwarnings("ignore", message="Hybrid functionals only support")

    warnings.filterwarnings("ignore", message="Use get_magnetic_symmetry()")
    warnings.filterwarnings("ignore", message="Use of properties is now deprecated")

    # avoid warning about selective_dynamics properties (can happen if user explicitly set "T T T" (or
    # otherwise) for the bulk):
    warnings.filterwarnings("ignore", message="Not all sites have property")

    # ignore warning about structure charge that appears when getting Vasprun.as_dict():
    warnings.filterwarnings("ignore", message="Structure charge")

    # ignore UFloat warning about std_dev==0 (from MP energy corrections), can potentially be removed in
    # future if/when this issue resolved upstream
    warnings.filterwarnings("ignore", message="Using UFloat objects with std_dev==0")


_ignore_pmg_warnings()


@cache
def warn_once(message: str, category: type[Warning] = UserWarning, key: Any = None) -> None:
    """
    Emit ``message`` as a warning at most once per unique ``(message, category,
    key)``, unless ``warn_once.cache_clear()`` is called.

    Unlike Python's default "show once per location" behaviour, this is
    immune to the ``__warningregistry__`` being reset whenever the warning
    filters are mutated (which dependencies such as ``pandas`` do internally,
    defeating the default dedup when warning repeatedly in a loop -- e.g. over
    temperatures/conditions).

    ``key`` is an optional `cheap` hashable used to check if the warning has
    already been called for a given object/situation; e.g. a |DefectEntry|
    ``name`` to warn once `per defect entry`.

    Used for the missing-degeneracy-factor warnings in ``doped.core``, which
    can otherwise be emitted many times in thermodynamic analysis loops.
    """
    warnings.warn(message, category, stacklevel=3)


class ParameterOrderWarning(FutureWarning):
    """
    Warning about the ``(bulk, defect)`` -> ``(defect, bulk)`` parameter
    ordering change for some functions in ``doped`` v4.0.

    TODO: Remove all parameter-order warning handling in v4.1.
    """


def _check_parameter_order_warning():
    """
    Check if the parameter order warning should be shown, based on the
    ``DOPED_WARN_PARAMETER_ORDER`` environment variable.

    Defaults to ``True`` if the environment variable is not set.
    """
    env = os.environ.get("DOPED_WARN_PARAMETER_ORDER")
    if env is None:
        return True
    return env.lower() not in ("0", "false", "no")


def _warn_parameter_order(func_name: str, stacklevel: int = 3):
    """
    Emit a ``ParameterOrderWarning`` for the given function name.
    """
    if not _check_parameter_order_warning():
        return
    warnings.warn(
        f"In doped v4.0, the parameter ordering for `{func_name}` was changed from "
        f"`(bulk_..., defect_..., ...)` to `(defect_..., bulk_..., ...)`. Please ensure your code uses "
        f"the correct ordering (and/or uses keyword arguments rather than positional arguments). This "
        f"warning can be disabled by setting the environment variable DOPED_WARN_PARAMETER_ORDER=false, "
        f"and will be removed in doped v4.1.",
        ParameterOrderWarning,
        stacklevel=stacklevel,
    )


def _doped_obj_properties_methods(obj):
    """
    Return a tuple of the attributes & properties and methods of a given
    object.

    Used in the ``__repr__()`` methods of ``doped`` objects.
    """
    attrs = {k for k in vars(obj) if not k.startswith("_")}
    methods = set()
    for k in dir(obj):
        with contextlib.suppress(Exception):
            if callable(getattr(obj, k)) and not k.startswith("_"):
                methods.add(k)
    properties = {name for name, value in inspect.getmembers(type(obj)) if isinstance(value, property)}
    return attrs | properties, methods


def get_mp_context():
    """
    Get a multiprocessing context that is compatible with the current OS.
    """
    try:
        return multiprocessing.get_context("forkserver")
    except ValueError:  # forkserver not available on Windows OS
        return multiprocessing.get_context("spawn")


def get_mp_processes(processes: int | None = None):
    """
    Get the number of processes to use with ``Pool``.
    """
    mp = get_mp_context()  # https://github.com/python/cpython/pull/100229
    return processes or max(1, mp.cpu_count() - 1)


@contextlib.contextmanager
def pool_manager(processes: int | None = None):
    r"""
    Context manager for ``multiprocessing`` ``Pool``, to throw a clearer error
    message when ``RuntimeError``\s are raised ``multiprocessing`` within
    ``doped`` is used in a python script.

    See the
    :ref:`Errors with Python Scripts <errors_with_python_scripts>` section.

    Args:
        processes (int | None):
            Number of processes to use with ``Pool``. If ``None``,
            will use ``mp.cpu_count() - 1`` (i.e. one less than the
            number of available CPUs).

    Yields:
        Pool:
            A ``Pool`` object with the specified number of processes.
    """
    pool = None
    try:
        mp = get_mp_context()  # https://github.com/python/cpython/pull/100229
        pool = mp.Pool(get_mp_processes(processes))
        yield pool
    except RuntimeError as orig_exc:
        if "freeze_support()" in str(orig_exc):
            raise RuntimeError(
                "When using doped in python scripts with multiprocessing (recommended), you must use the "
                "`if __name__ == '__main__':` syntax, see "
                "https://doped.readthedocs.io/en/latest/Troubleshooting.html#errors-with-python-scripts "
                "-- alternatively you can set processes=1 (but this will be slower)"
            ) from orig_exc
        raise orig_exc
    finally:
        if pool is not None:
            pool.close()
            pool.join()
