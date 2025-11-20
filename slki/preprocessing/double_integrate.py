# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
from typing import Any

import numpy as np
from typing_extensions import override

from ..config import Config
from ..utils import dict_to_object_str
from .external_utils import double_integration
from .stage import Stage


class DoubleIntegrate(Stage):
    """
    A preprocessing stage that applies double integration to one or more time series.

    This class provides methods to perform double integration on time series data,
    which involves twice integrating each signal with respect to time.
    """

    SUPPORTS_ONLY_ITERATIVE: bool = True

    def run(
        self,
        cutoff_freq: tuple[float, float] = (0.5, 15),
        tap_s: float = 3,
        f_order: int = 2,
        convert_m_to_mm: bool = True,
        one_side_taper_flipped: bool = False,
        neutral_element: float = 0,
    ) -> None:
        """
        Runs the double integration stage.

        This method applies a custom filter to smooth and integrate each signal.
        It also allows for conversion of units from meters to millimeters, and flipping of the one-sided taper.

        Args:
            cutoff_freq (Tuple[float, float], optional): A tuple of minimum and maximum frequency values used
                for filtering. Defaults to (0.5, 15).
            tap_s (float | int, optional): The value of Tap-S used in the Savitzky-Golay filter. Defaults to 3.
            f_order (int, optional): The order of the Savitzky-Golay filter. Defaults to 2.
            convert_m_to_mm (bool, optional): Whether to convert units from meters to millimeters. Defaults to True.
            one_side_taper_flipped (bool, optional): Whether to flip the one-sided taper used in the filtering
                process. Defaults to False.
            neutral_element (int | float, optional): The value used to replace leading and trailing segments with
                low signal quality. Defaults to 0.
        """
        self._cutoff_freq = cutoff_freq
        self._tap_s = tap_s
        self._f_order = f_order
        self._convert_m_to_mm = convert_m_to_mm
        self._one_side_taper_flipped = one_side_taper_flipped
        self._neutral_element = neutral_element

        self.apply(self._double_integrate, "Double Integration")

    @override
    def run_default(self) -> None:
        self.run(
            cutoff_freq=Config.CUTOFF_FREQ_IN_HZ,
            tap_s=Config.TAP_S,
            f_order=Config.F_ORDER,
            convert_m_to_mm=Config.CONVERT_M_TO_MM,
            one_side_taper_flipped=Config.ONE_SIDE_TAPER_FLIPPED,
        )

    def _double_integrate(self, data: np.ndarray, metadata: list[dict[str, Any]], **kwargs) -> np.ndarray:
        """
        Applies double integration to a single signal (one time series).

        This method uses a custom filter to smooth and integrate a signal.
        It also allows for conversion of units from meters to millimeters, and flipping of the one-sided taper.

        Args:
            data (np.ndarray): A 2D NumPy array containing a single row representing a single time series.
            metadata (List[Dict[str, Any]]): A list of dictionaries containing metadata for each time series.
            **kwargs: Additional arguments will be ignored.

        Returns:
            np.ndarray: The 2D input `data` with double integration applied to the signal.
        """
        signal_data, meta = data.squeeze(axis=0), metadata[0]

        # get sample rate
        sr = meta.get("sample_rate_in_hz", None)
        if sr is None:
            origin: dict[str, Any] = meta.get("origin", None)  # type: ignore[assignment]
            origin_str = dict_to_object_str(origin, "Origin") or '"unknown origin"'
            self.logger.error(f"No sample rate found for {origin_str}. Skip denoising for this time series.")
            return data

        # double integration
        signal_length = len(signal_data) if self.data_container.resampled else meta["sample_length"]
        signal = signal_data[:signal_length]
        signal_data[:signal_length] = double_integration(
            signal,
            sr,
            self._tap_s,
            self._cutoff_freq,
            self._f_order,
            start_tapering_only=self._one_side_taper_flipped,
            perform_on_flipped_data=self._one_side_taper_flipped,
        )
        signal_data[signal_length:] = self._neutral_element

        # convert to mm
        if self._convert_m_to_mm:
            signal_data[:signal_length] *= 1000

        return np.expand_dims(signal_data, axis=0)
