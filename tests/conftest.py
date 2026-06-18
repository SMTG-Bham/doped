"""
Shared ``pytest`` configuration/fixtures for the ``doped`` test suite.
"""

import pytest

from doped.core import warn_once


@pytest.fixture(autouse=True)
def _clear_warn_once_cache():
    """
    Reset the ``warn_once`` dedup cache before each test (to avoid warning
    suppression from previously-run tests).
    """
    warn_once.cache_clear()
