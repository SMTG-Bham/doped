"""
Shared ``pytest`` configuration/fixtures for the ``doped`` test suite.
"""

import pytest

from doped.core import warn_once
from doped.utils import _ignore_pmg_warnings
from doped.utils.plotting import _resolve_doped_style


@pytest.fixture(scope="session", autouse=True)
def _install_custom_font():
    """
    Install ``doped``'s custom plotting font (if not already present) at the
    start of the test session.

    Otherwise, in fresh environments (e.g. CI), the font installation messages
    are printed by the first plotting call of the session, breaking any
    ``stdout`` checks in whichever test that happens to be.
    """
    _resolve_doped_style()


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
    ``warnings.catch_warnings()``, so filters added by imports during
    collection are discarded before tests run. This function-scoped fixture
    runs inside each test's own ``catch_warnings`` context, so the filters are
    in place for the test (and inherited by any ``catch_warnings`` blocks
    within it).
    """
    _ignore_pmg_warnings()
