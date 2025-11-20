# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
import numpy as np
from tsmoothie.smoother import ConvolutionSmoother, ExponentialSmoother
from typing_extensions import override

from ..config import Config, TSmoothType
from .stage import Stage


class Smooth(Stage):
    """A preprocessing stage that smooths one or more time series."""

    # TODO: check if `self._smooth` works as matrix
    SUPPORTS_ONLY_ITERATIVE: bool = True

    def run(self, which: TSmoothType = "conv", **smoother_kwargs) -> None:
        """
        Runs the smoothing stage.

        This method applies a specified smoothing method to each time series in the input data
        using the provided kwargs.

        Args:
            which (TSmoothType): The type of smoother to use. Defaults to "conv" (convolution).
            **smoother_kwargs: Additional keyword arguments for the smoother.
        """
        self._smoother = self._get_smoother(which, **smoother_kwargs)
        self.apply(self._smooth, "Smoothing")

    @override
    def run_default(self) -> None:
        self.run(Config.SMOOTH_TYPE, **Config.SMOOTHER_KWARGS)

    def _get_smoother(self, which: TSmoothType, **smoother_kwargs):
        """
        Returns a smoother object based on the specified type.

        Args:
            which (TSmoothType): The type of smoother to use.
            **smoother_kwargs: Additional keyword arguments for the smoother.

        Raises:
            ValueError: If an unsupported smoother type is requested.

        Returns:
            Callable[[np.ndarray], np.ndarray]: A callable instance of the selected smoother.
        """
        if which == "conv":
            kwargs = dict(window_len=10, window_type="hanning")
            kwargs.update(smoother_kwargs)
            return ConvolutionSmoother(**kwargs)
        if which == "exp":
            kwargs = dict(window_len=10, alpha=0.5)
            kwargs.update(smoother_kwargs)
            return ExponentialSmoother(**kwargs)
        raise ValueError(f"{which} smoother is not defined")

    def _smooth(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Smooths a single signal (time series) using a specified smoother.

        This method applies the smoothing method to the input data and returns the smoothed result.

        Args:
            data (np.ndarray): A 2D NumPy array containing a single row representing a single time series.
            **kwargs: Additional arguments will be ignored.

        Returns:
            np.ndarray: The smoothed time series data (as 2D NumPy array).
        """
        return self._smoother.smooth(data).smooth_data
