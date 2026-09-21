"""
Calculator-agnostic utilities for handling calculation input/output files;
currently for locating calculation output files and folders.

These are generic algorithms parameterised by calculator-specific filename
masks and subfolder priority lists, which are supplied by each
``doped.io.<calculator>`` backend (see e.g. their usage in
``doped.io.vasp.outputs``).
"""

import os
import re
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from pymatgen.util.typing import PathLike


def find_archived_fname(fname, raise_error=True):
    """
    Find a suitable filename, taking account of possible use of compression
    software.
    """
    if os.path.exists(fname):
        return fname
    # Check for archive files
    for ext in [".gz", ".xz", ".bz", ".lzma"]:
        if os.path.exists(fname + ext):
            return fname + ext
    if raise_error:
        raise FileNotFoundError
    return None


def _get_output_files_and_check_if_multiple(
    output_file: PathLike, path: PathLike = ".", search_patterns: list[str] | None = None
) -> tuple[PathLike, bool]:
    """
    Search for all files with filenames matching ``output_file``, case-
    insensitive.

    Args:
        output_file (PathLike):
            The filename to search for (case-insensitive).
        path (PathLike):
            The path to the directory to search in.
        search_patterns (list[str]):
            Patterns which must `all` appear in a (lower-cased) filename for
            it to match. Defaults to ``[output_file.lower()]``; calculator
            backends can supply looser patterns (e.g. ``["vasprun", ".xml"]``
            for ``vasprun.xml`` with VASP).

    Returns:
        Tuple[PathLike, bool]:
            The path to the identified file, and a boolean indicating whether
            multiple files were found.
    """
    if search_patterns is None:
        search_patterns = [str(output_file).lower()]

    files = os.listdir(path)
    output_files = [
        filename
        for filename in files
        if all(i in filename.lower() for i in search_patterns) and not filename.startswith(".")
    ]
    # sort by direct match to {output_file}, direct match to {output_file}.gz, then alphabetically:
    if output_files := sorted(
        output_files,
        key=lambda x: (x == output_file, x == f"{output_file}.gz", x),
        reverse=True,
    ):
        output_path = os.path.join(path, output_files[0])
        return (output_path, True) if len(output_files) > 1 else (output_path, False)
    return (
        os.path.join(path, output_file),
        False,
    )  # so `get_X()` will raise an informative FileNotFoundError


def _dataframe_of_files(root: Path) -> pd.DataFrame:
    """
    Get a dataframe with one row per file under ``root``, found recursively.

    Hidden files/directories (names starting with ``"."``) and files
    sitting directly in ``root`` (i.e. with only one path component
    relative to ``root``) are excluded.

    Columns: ``filename``, ``full_path``, ``folder_path``,
    ``folder_in_root`` (the first path component relative to ``root``).

    Args:
        root (Path):
            Path to the root directory.

    Returns:
        pd.DataFrame:
            One row per discovered file.
    """
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for f in root.rglob("*"):  # recursively find all files under root, ignoring hidden folders/files
        if f.is_file():
            relative_parts = f.relative_to(root).parts
            if any(part.startswith(".") for part in relative_parts) or len(relative_parts) < 2:
                continue  # ignore hidden files and folders, and files in root directory itself
            rows.append(
                {
                    "filename": f.name,
                    "full_path": f,
                    "folder_path": f.parent,
                    "folder_in_root": relative_parts[0],
                }
            )
    return pd.DataFrame(rows)


def _get_calc_files_df(root: Path, calc_output_mask: Iterable[str]) -> pd.DataFrame:
    """
    Get a DataFrame of calculation output files (matching ``calc_output_mask``)
    found recursively under ``root``, excluding hidden files/directories and
    files sitting directly in ``root``.

    This is a filtered view of :func:`_dataframe_of_files`.

    Args:
        root (Path):
            Path to the root directory.
        calc_output_mask (Iterable[str]):
            Iterable of filename patterns identifying calculation output
            files for the calculator used (e.g. ``("vasprun.xml",
            "vasprun.xml.gz")`` for VASP). Matching is case-insensitive.

    Returns:
        pd.DataFrame:
            One row per matching calculation output file.
    """
    files_df = _dataframe_of_files(root)
    if files_df.empty:
        return pd.DataFrame()
    pattern = "|".join(map(re.escape, calc_output_mask))
    return files_df[files_df["filename"].str.contains(pattern, regex=True, na=False)]


def _determine_subfolder(
    files_df: pd.DataFrame,
    candidate_folders: list[str] | None = None,
    *,
    subfolder_priority: list[str],
) -> str:
    """
    Pick the highest-priority calculation subfolder name present in
    ``files_df`` (restricted to rows whose ``folder_in_root`` is in
    ``candidate_folders``), or ``"."`` if none of the priority names are found.

    Args:
        files_df (pd.DataFrame):
            DataFrame produced by :func:`_dataframe_of_files`, filtered
            to calculation output files.
        candidate_folders (list[str]):
            Top-level folder names to consider. If ``None`` (default),
            considers all top-level folder names.
        subfolder_priority (list[str]):
            Priority order for subfolder names for the calculator used
            (e.g. ``["vasp_ncl", ..., "vasp_gam"]`` for VASP), compared
            case-insensitively.

    Returns:
        str:
            The detected subfolder name, or ``"."``.
    """
    candidate_folder_df = (
        files_df[files_df["folder_in_root"].isin(candidate_folders)]
        if candidate_folders is not None
        else files_df
    )
    for subfolder in subfolder_priority:
        if any(subfolder in p.name.lower() for p in candidate_folder_df["folder_path"].unique()):
            return subfolder
    return "."


def _find_calc_outputs(
    output_path: PathLike,
    subfolder: PathLike | None = None,
    *,
    calc_output_mask: Iterable[str],
    subfolder_priority: list[str],
) -> tuple[pd.DataFrame, list[str], str]:
    """
    Recursively find calculation output files under ``output_path`` and auto-
    detect the calculation subfolder when ``subfolder`` is ``None``.

    Shared discovery logic used by both :func:`~doped.parsing.DefectsParser`
    and :meth:`~doped.chemical_potentials.CompetingPhasesAnalyzer`.

    Args:
        output_path (PathLike):
            Root directory to search.
        subfolder (PathLike | None):
            Explicit subfolder name (e.g. ``"vasp_std"``). If ``None``,
            auto-detected using ``subfolder_priority``.
        calc_output_mask (Iterable[str]):
            Filename patterns identifying calculation output files for the
            calculator used; see :func:`_get_calc_files_df`.
        subfolder_priority (list[str]):
            Priority order for subfolder names for the calculator used; see
            :func:`_determine_subfolder`.

    Returns:
        tuple[pd.DataFrame, list[str], str]:
            ``(calc_files_df, candidate_folders, resolved_subfolder)``
            where *resolved_subfolder* is ``"."`` when no priority
            subfolder is found.
    """
    calc_files_df = _get_calc_files_df(Path(output_path), calc_output_mask)
    if calc_files_df.empty:
        return pd.DataFrame(), [], "."

    candidate_folders = calc_files_df["folder_in_root"].unique().tolist()
    resolved_subfolder = (
        _determine_subfolder(calc_files_df, candidate_folders, subfolder_priority=subfolder_priority)
        if subfolder is None
        else str(subfolder)
    )
    return calc_files_df, candidate_folders, resolved_subfolder


def _multiple_files_warning(file_type, directory, chosen_filepath, action, dir_type="bulk"):
    """
    Warn that multiple files matching ``file_type`` were found in
    ``directory``, and that ``chosen_filepath`` is being used to ``action``.
    """
    filename = os.path.basename(chosen_filepath)
    warnings.warn(
        f"Multiple `{file_type}` files found in {dir_type} directory: {directory}. Using {filename} to "
        f"{action}"
    )
