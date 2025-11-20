# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
import numpy as np

from .stage import Stage


class Absolute(Stage):
    """
    A stage that applies the absolute value function to the input data.

    This stage uses NumPy's `abs` function to compute the absolute values of each data element.
    """

    def run(self) -> None:
        """Applies the absolute value function to the input data."""
        self.apply(np.abs, "Absolute")
