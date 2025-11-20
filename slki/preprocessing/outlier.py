# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
import numpy as np
from typing_extensions import override

from ..config import Config
from .stage import Stage


class Outlier(Stage):
    """A preprocessing stage that reduces outliers from one or more time series."""

    def run(self, reduction_itervals: int) -> None:
        """Runs the outlier reduction stage."""
        assert reduction_itervals >= 0
        self._reduction_itervals = reduction_itervals
        self.apply(self._reduce_outlier, "Outlier reducing")

    @override
    def run_default(self) -> None:
        self.run(Config.OUTLIER_REDUCTION_INTERVALS)

    def _reduce_outlier(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Reduce outliers from one or more time series using the natural logarithm (element-wise) multiple times.

        Args:
            data (np.ndarray): The input time series data.
            **kwargs: Additional arguments will be ignored.

        Returns:
            np.ndarray: The input data with outliers reduced.
        """
        for _ in range(self._reduction_itervals):
            data[data > 0.0] = np.log(data[data > 0.0] + 1)
            data[data < 0.0] = -np.log(-data[data < 0.0] + 1)
        return data
