# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""This module provides utility functions for customizing progress bars using the `tqdm` library."""

from typing import Any

from ..config import Config


def get_tqdm_kwargs(desc: str | None, enable: bool = True, **tqdm_overwrite_kwargs) -> dict[str, Any]:
    """
    Get expected kwargs for `tqdm` to customize a progress bar.

    Args:
        desc (Optional[str]): A description string for the progress bar.
        enable (bool, optional): Whether to disable or enable the progress bar. Defaults to True.
        **tqdm_overwrite_kwargs: Additional kwargs to overwrite default values from Config.TQDM_OPTIONS.

    Returns:
        Dict[str, Any]: Expected kwargs for `tqdm` to customize a progress bar.
    """
    tqdm_kwargs = Config.TQDM_OPTIONS.copy()
    tqdm_kwargs.update(tqdm_overwrite_kwargs or {})
    tqdm_kwargs["desc"] = desc
    tqdm_kwargs["disable"] = tqdm_kwargs.get("disable", False) or not enable
    return tqdm_kwargs


def get_tqdm_thread_map_kwargs(desc: str | None, enable: bool = True, **tqdm_overwrite_kwargs) -> dict[str, Any]:
    """
    Get expected kwargs for `tqdm` to customize a parallel working progress bar.

    Args:
        desc (Optional[str]): A description string for the progress bar.
        enable (bool, optional): Whether to disable or enable the progress bar. Defaults to True.
        **tqdm_overwrite_kwargs: Additional kwargs to overwrite default values from Config.TQDM_OPTIONS.

    Returns:
        Dict[str, Any]: Expected kwargs for `tqdm` to customize a parallel working progress bar.
    """
    tqdm_kwargs = get_tqdm_kwargs(desc, enable, **tqdm_overwrite_kwargs)
    if Config.TQDM_MAX_WORKERS is not None:
        tqdm_kwargs["max_workers"] = Config.TQDM_MAX_WORKERS
    return tqdm_kwargs
