"""
Utility functions for ``doped`` tests, which are used in multiple test modules.
"""

import os
import shutil
import sys
import warnings
from functools import wraps
from io import StringIO

import numpy as np
import pytest

# for pytest-mpl:
module_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(module_path, "data")
EXAMPLE_DIR = os.path.join(module_path, "../examples")
BASELINE_DIR = f"{data_dir}/remote_baseline_plots"
STYLE = f"{module_path}/../doped/utils/doped.mplstyle"


def custom_mpl_image_compare(filename, style=STYLE, **kwargs):
    """
    Set our default settings for MPL image compare.
    """

    def decorator(func):
        @wraps(func)
        @pytest.mark.mpl_image_compare(
            baseline_dir=BASELINE_DIR,
            filename=filename,
            style=style,
            savefig_kwargs={"transparent": True, "bbox_inches": "tight"},
            **kwargs,
        )
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _print_warning_info(warnings_list):
    if not warnings_list:
        return

    for i, warn in enumerate(warnings_list, 1):
        # warn is a warnings.WarningMessage
        print(f"\n--- Warning {i}/{len(warnings_list)} ---")
        print(f"Category: {warn.category.__name__}")
        print(f"Message : {warn.message}")
        print(f"Origin  : {warn.filename}:{warn.lineno}")


def plot_chempot_heatmap_and_test_no_warnings(cpa_or_defect_thermo, **kwargs):
    """
    Plot chemical potential heatmap from a ``CompetingPhasesAnalyzer`` or
    ``DefectThermodynamics`` object and assert no warnings are raised.
    """
    with warnings.catch_warnings(record=True) as w:
        plot = cpa_or_defect_thermo.plot_chempot_heatmap(**kwargs)
    _print_warning_info(w)
    assert not w
    return plot


def if_present_rm(path):
    """
    Remove file or directory if it exists.
    """
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def _run_func_and_capture_stdout_warnings(func, *args, **kwargs):
    original_stdout = sys.stdout  # Save a reference to the original standard output
    sys.stdout = StringIO()  # Redirect standard output to a stringIO object.
    w = None
    try:
        with warnings.catch_warnings(record=True) as w:
            result = func(*args, **kwargs)
        output = sys.stdout.getvalue()  # Return a str containing the printed output
    finally:
        sys.stdout = original_stdout  # Reset standard output to its original value.

    print(f"Running {func.__name__} with args: {args} and kwargs: {kwargs}:")
    print(output)
    if w:
        print(f"Warnings ({len(w)}):")
        _print_warning_info(w)
    print(f"Result: {result}\n")

    return result, output, w


def _potcars_available() -> bool:
    """
    Check if the POTCARs are available for the tests (i.e. testing locally).

    If not (testing on GitHub Actions), POTCAR testing will be skipped.
    """
    from doped.vasp import _test_potcar_functional_choice

    try:
        _test_potcar_functional_choice("PBE")
        return True
    except ValueError:
        return False


def _compare_prim_interstitial_coords(result, expected):
    """
    Check that prim_interstitial_coords_mult_and_equiv_coords attribute values
    match.
    """
    if result is None:
        assert expected is None
        return

    assert len(result) == len(expected), "Lengths do not match"

    for (r_coord, r_num, r_list), (e_coord, e_num, e_list) in zip(result, expected, strict=False):
        assert np.array_equal(r_coord, e_coord), "Coordinates do not match"
        assert r_num == e_num, "Number of coordinates do not match"
        assert all(
            np.array_equal(r, e) for r, e in zip(r_list, e_list, strict=False)
        ), "List of arrays do not match"


def _compare_attributes(obj1, obj2, exclude=None):
    """
    Check that two objects are equal by comparing their public
    attributes/properties.

    Handles special cases for ``DefectsGenerator``
    (``prim_interstitial_coords_mult_and_equiv_coords``)
    and ``DefectThermodynamics`` (``bulk_dos``) objects.
    """
    if exclude is None:
        exclude = set()  # Create an empty set if no exclusions

    for attr in dir(obj1):
        if attr.startswith("_") or attr in exclude or callable(getattr(obj1, attr)):
            continue  # Skip private, excluded, and callable attributes

        print(attr)
        val1 = getattr(obj1, attr)
        val2 = getattr(obj2, attr)

        if isinstance(val1, np.ndarray):
            assert np.allclose(val1, val2)
        elif attr == "prim_interstitial_coords_mult_and_equiv_coords":
            _compare_prim_interstitial_coords(val1, val2)
        elif attr == "defects" and any(len(i.defect_structure) == 0 for i in val1["vacancies"]):
            continue  # StructureMatcher comparison breaks for empty structures, which we can have with
            # our 1-atom primitive Cu input
        elif attr == "bulk_dos" and val1 is not None:
            assert val1.as_dict() == val2.as_dict()
        elif isinstance(val1, list | tuple) and all(isinstance(i, np.ndarray) for i in val1):
            assert all(
                np.array_equal(i, j) for i, j in zip(val1, val2, strict=False)
            ), "List of arrays do not match"
        else:
            assert val1 == val2


def assert_df_rows_equal(df, expected_rows, check_len=False):
    """
    Assert that a ``DataFrame``'s rows match the expected values.

    Handles float comparisons with ``np.isclose`` and exact comparisons
    for other types.

    Args:
        df: ``pandas`` ``DataFrame`` to check.
        expected_rows: List of lists containing expected row values.
        check_len: If ``True``, also verify the number of rows matches.
            Default is ``False``.
    """
    if check_len:
        assert len(df) == len(
            expected_rows
        ), f"DataFrame has {len(df)} rows, expected {len(expected_rows)}"

    for i, row in enumerate(expected_rows):
        for ii, entry in enumerate(row):
            actual = df.iloc[i, ii]
            if isinstance(entry, float):
                assert np.isclose(
                    actual, entry
                ), f"Row {i}, column {ii}: {actual} != {entry} (float comparison)"
            else:
                assert actual == entry, f"Row {i}, column {ii}: {actual} != {entry}"
