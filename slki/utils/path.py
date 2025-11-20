# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""This module provides utility functions for working with file paths."""

from collections.abc import Generator, Sequence
from glob import glob
import logging
import mimetypes
import os
from pathlib import Path


def is_file_readable(filepath: str | Path, logger: logging.Logger | None = logging.root) -> bool:
    """
    Check if a file is readable.

    Args:
        filepath (str | Path): The path to the file.
        logger (Optional[logging.Logger], optional): The logger to use for logging warnings. Defaults to logging.root.

    Returns:
        bool: True if the file is readable, False otherwise.
    """
    filepath = str(filepath)
    if not os.path.isfile(filepath):
        if logger:
            logger.warning(f"Path '{filepath}' is not a file or does not exist. Skip.")
        return False
    if not os.access(filepath, os.R_OK):
        if logger:
            logger.warning(f"File '{filepath}' is not readable. Skip.")
        return False
    return True


def are_files_readable(filepaths: Sequence[str | Path], logger: logging.Logger | None = logging.root) -> bool:
    """
    Check if all files in the given list of filepaths are readable.

    Args:
        filepaths (Sequence[str | Path]): A sequence of filepaths to check.
        logger (Optional[logging.Logger], optional): The logger to use for logging warnings. Defaults to logging.root.

    Returns:
        bool: True if all files are readable, False otherwise.
    """
    return all(is_file_readable(filepath, logger) for filepath in filepaths)


def find_import_files(
    path_pattern: str | Sequence[str],
    supported_file_exts: Sequence[str] = [],
    supported_minetypes: Sequence[str] = [],
    logger: logging.Logger = logging.root,
) -> Generator[str, None, None]:
    """
    Find import files based on the given path pattern(s) and filter criteria.

    Args:
        path_pattern (str | Sequence[str]): The path pattern(s) to search for import files.
        supported_file_exts (Sequence[str], optional): List of supported file extensions to filter the files. Defaults to an empty list.
        supported_minetypes (Sequence[str], optional): List of supported mimetypes to filter the files. Defaults to an empty list.
        logger (logging.Logger, optional): The logger to use for logging warnings. Defaults to logging.root.

    Yields:
        str: The next import file path.
    """  # noqa: E501
    if isinstance(path_pattern, str):
        path_pattern = [path_pattern]

    for pattern in path_pattern:
        paths = glob(pattern)
        if not paths:
            logger.warning(f"No file found for '{pattern}'. Skip.")
            continue

        for path in paths:
            if os.path.isdir(path):
                for filepath in Path(path).glob("*"):
                    if (
                        (not supported_file_exts and not supported_minetypes)
                        or Path(filepath).suffix in supported_file_exts
                        or mimetypes.guess_type(filepath)[0] in supported_minetypes
                    ) and is_file_readable(filepath, logger):
                        yield str(filepath)
            elif is_file_readable(path, logger):
                yield path
