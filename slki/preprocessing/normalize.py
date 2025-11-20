# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
from collections.abc import Callable

import numpy as np
import sklearn.preprocessing
import tslearn.preprocessing
from typing_extensions import override

from ..config import Config, TNorm
from .stage import Stage


class Normalize(Stage):
    """
    A preprocessing stage that normalizes one or more time series.

    This class provides methods to scale and shift time series data,
    typically used for feature scaling in machine learning pipelines.
    """

    def run(self, norm: TNorm = "mone_one_zero_fix") -> None:
        """
        Runs the normalization stage.

        This method applies feature scaling and shifting to one or more time series,
        using a specified normalization scheme.

        Args:
            norm (TNorm, optional): The normalization scheme to use. Defaults to "mone_one_zero_fix".
        """
        self._scaler = self._get_normalizer(norm)
        self._norm = norm
        self.apply(self._normalize, "Normalizing")

    @override
    def run_default(self) -> None:
        self.run(Config.NORM_TYPE)

    def _normalize(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Applies normalization to one or more time series.

        This method scales and shifts the input data using a specified normalization scheme,
        and removes any additional dimension created by the scaling process.

        Args:
            data (np.ndarray): The time series data to be normalized.
            **kwargs: Additional arguments will be ignored.

        Returns:
            np.ndarray: The normalized time series data.
        """
        # normalize data
        data = self._scaler(data)

        # TimeSeriesScalerXXX transformers create an additional dimension which have to be remove
        if self._norm in ["mean_var", "zero_one", "mone_one"]:
            data = data.squeeze(axis=-1)

        return data

    def _get_normalizer(self, norm: TNorm) -> Callable[[np.ndarray], np.ndarray]:
        """
        Returns a normalization transformer based on the specified scheme.

        This method selects and returns an instance of a time series scaler or normalizer,
        depending on the requested normalization scheme.

        Args:
            norm (TNorm): The normalization scheme to use. Supported schemes are:

                - "mean_var": mean and variance scaling
                - "zero_one": zero-one scaling
                - "mone_one": (-1, 1) scaling
                - "l1", "l2", or "max": L1, L2, or max norm regularization
                - "mone_one_zero_fix": a custom normalization scheme that scales the data
                  between (-1, 1) while keeping zero at zero

        Raises:
            ValueError: If an unsupported normalization scheme is requested.

        Returns:
            Callable[[np.ndarray], np.ndarray]: A callable instance of the selected normalization transformer.
        """
        if norm == "mean_var":
            return tslearn.preprocessing.TimeSeriesScalerMeanVariance().fit_transform
        if norm == "zero_one":
            return tslearn.preprocessing.TimeSeriesScalerMinMax().fit_transform
        if norm == "mone_one":
            return tslearn.preprocessing.TimeSeriesScalerMinMax(value_range=(-1, 1)).fit_transform
        if norm == "l1" or norm == "l2" or norm == "max":
            return sklearn.preprocessing.Normalizer(norm=norm).fit_transform
        if norm == "mone_one_zero_fix":
            return self._norm_plus_minus_one_but_keep_zero_at_zero
        raise ValueError(f"{norm} normalization is not defined")

    def _norm_plus_minus_one_but_keep_zero_at_zero(self, data: np.ndarray) -> np.ndarray:
        factors = np.max(np.abs(data), axis=-1)
        factors = factors[:, None]
        factors[factors == 0] = 1  # avoid division by zero (but signals with all zeros should not exists)
        data = data / factors
        return data
