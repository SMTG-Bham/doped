"""
Shared ``pytest`` configuration/fixtures for the ``doped`` test suite.
"""

import pytest

from doped.core import warn_once
from doped.utils import _ignore_pmg_warnings


@pytest.fixture(autouse=True)
def _clear_warn_once_cache():
    """
    Reset the ``warn_once`` dedup cache before each test (to avoid warning
    suppression from previously-run tests).
    """
    warn_once.cache_clear()


@pytest.fixture(autouse=True)
def _apply_doped_warning_filters():
    """
    Re-apply ``doped``'s ``pymatgen`` noise filters for each test.

    ``doped.utils`` applies these at import, but ``pytest`` wraps collection in
    ``warnings.catch_warnings()``, so filters added by imports during collection
    are discarded before tests run. This function-scoped fixture runs inside
    each test's own ``catch_warnings`` context, so the filters are in place for
    the test (and inherited by any ``catch_warnings`` blocks within it).
    """
    _ignore_pmg_warnings()
