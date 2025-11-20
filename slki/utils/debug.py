# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""This module provides utility functions for debugging purposes."""

import random

import numpy as np


def ensure_deterministic(seed: int | None = None) -> None:
    """
    Ensure deterministic behavior.

    Manipulates the random number generators to ensure deterministic behavior.
    If the seed is not set or `None`, the function does nothing.

    Args:
        seed (Optional[int], optional): Manual seed value. Defaults to None.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
