"""
Shared pytest configuration for doped tests.
"""

import os

# Suppress ParameterOrderWarning in tests (internal calls use the correct ordering).
# TODO: Remove in doped v4.1 (along with ParameterOrderWarning itself, and this file entirely).
os.environ["DOPED_WARN_PARAMETER_ORDER"] = "false"
