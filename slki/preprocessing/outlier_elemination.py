# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..utils.peak import detect_peaks
from .stage import Stage


class OutlierElimination(Stage):
    """A preprocessing stage that removes all outliers from one or more time series."""

    SUPPORTS_ONLY_ITERATIVE: bool = True

    def run(self) -> None:
        """Runs the outlier elimination stage."""
        self.apply(self._eliminate_outlier, "Outlier eliminating")

    def _eliminate_outlier(self, data: np.ndarray, metadata: list[dict[str, Any]], **kwargs) -> np.ndarray:
        """
        Eliminate outliers from one time series using the natural logarithm (element-wise) multiple times.

        Args:
            data (np.ndarray): The input signal data.
            metadata (List[Dict[str, Any]]): Metadata associated with the input signal.
            **kwargs: Additional arguments will be ignored.

        Returns:
            np.ndarray: The input data with outliers removed.
        """
        signal, meta = data.squeeze(axis=0), metadata[0]
        sample_length = meta["sample_length"]
        while self._has_outliers(signal, sample_length):
            signal[signal > 0.0] = np.log(signal[signal > 0.0] + 1)
            signal[signal < 0.0] = -np.log(-signal[signal < 0.0] + 1)
        return np.expand_dims(signal, axis=0)

    def _has_outliers(self, sample: np.ndarray, sample_length: int) -> bool:
        """
        Check if the signal contains outliers.

        Args:
            sample (np.ndarray): The signal data.
            sample_length (int): The original signal length.

        Returns:
            bool: True is outliers were detected; otherwise False.
        """
        peaks = detect_peaks(sample, sample_length)
        sample_peaks = sample[peaks]
        bxpstats = plt.cbook.boxplot_stats(sample_peaks)
        fliers = bxpstats[0]["fliers"]
        return bool(fliers)
