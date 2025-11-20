# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
Shows the calculation of displacements from acceleration on exemplary DB data.

The source code from this file is in its initial version was taken from a jupiter notebook
created by Susanne Reetz (reet_su) from June 11th 2024.
The notebook was provided by Dr. rer. nat. Markus Lange (lang_m25).

License: Use only within the projects "Digitiale Weich 2.0" und "SLKI", according to the
         respective agreements between DLR and BREUER and/or for DLR-internal purposes.
"""

from typing import Literal

import numpy as np
from scipy import integrate, signal


def butter_filter(
    y: np.ndarray,
    sampling_rate_hz: float,
    cutoff_freq_hz: np.typing.ArrayLike,
    order: int,
    f_type: Literal["lowpass", "highpass", "bandpass", "bandstop"],
) -> np.ndarray:
    """
    Applies a forward- and backward butterworth filter to the signal and returns it.

    Args:
        y (np.ndarray): 1d signal array
        sampling_rate_hz (float): sampling rate of signal array in Hz
        cutoff_freq_hz (np.typing.ArrayLike): cutoff filter frequencies in Hz
            For lowpass and highpass filters, cutoff_frequencies is a scalar;
            for bandpass and bandstop filters, cutoff_frequencies is a `length-2` sequence.
        order (int): filter order
        f_type (Literal["lowpass", "highpass", "bandpass", "bandstop"]): type of filter

    Returns:
        np.ndarray: filtered signal array
    """
    # design filter
    sos = signal.butter(  # type: ignore[call-overload]
        N=order,
        Wn=cutoff_freq_hz,
        btype=f_type,
        analog=False,
        output="sos",
        fs=sampling_rate_hz,
    )
    # filter signal forward and backwards
    return signal.sosfiltfilt(x=y, sos=sos)


def tapering(y: np.ndarray, sampling_rate_hz: float, tap_s: float, start_only: bool = False) -> np.ndarray:
    """
    Tapers signal with tukey window at beginning and end and returns it.

    Args:
        y (np.ndarray): 1d signal array
        sampling_rate_hz (float): sampling rate of signal array in Hz
        tap_s (float): number of seconds to taper the signal array at the beginning and end
        start_only (bool): whether to tapers signal with tukey window at beginning only

    Returns:
        np.ndarray: tapered signal array
    """
    window = signal.windows.tukey(len(y), alpha=2 * tap_s * sampling_rate_hz / len(y))
    if start_only:
        window[int(len(window) / 2) :] = 1
    return np.transpose(window * np.transpose(y))


def double_integration(
    y: np.ndarray,
    sampling_rate_hz: float,
    tap_s: float,
    cutoff_freq_hz: tuple,
    f_order: int,
    *,
    start_tapering_only: bool = False,
    perform_on_flipped_data: bool = False,
) -> np.ndarray:
    """
    Performs a double integration of the filtered and tapered signal.

    If signal has the unit `m/s^2`, then the double integrated signal has the unit `m`.

    Args:
        y (np.ndarray): 1d signal array
        sampling_rate_hz (float): sampling rate of signal array in Hz
        tap_s (float): number of seconds to taper the data array at the beginning and end
        cutoff_freq_hz (tuple): cutoff filter frequencies in Hz (low_hz, high_hz)
        f_order (int): filter order
        start_tapering_only (bool): whether to tapers signal with tukey window at beginning only. Default is False.
        perform_on_flipped_data (bool): whether to flip the data before filtering, tapering and double integration.
            Default is False.

    Returns:
        np.ndarray: double intergrated signal
    """
    if perform_on_flipped_data:
        # flip array to better handle measurements starting in the middle of a train passage
        # (i.e. beginning of train passage missing)
        y = np.flip(y)

    # integrate on a filtered and tapered signal TWICE
    for _ in range(2):
        # remove mean and trend
        signal.detrend(y, type="linear", overwrite_data=True)
        # tapering
        y = tapering(y, sampling_rate_hz, tap_s, start_only=start_tapering_only)
        # bandpass filter
        y = butter_filter(y, sampling_rate_hz, cutoff_freq_hz, f_order, "bandpass")
        # tapering
        y = tapering(y, sampling_rate_hz, tap_s, start_only=start_tapering_only)
        # integration
        y = integrate.cumulative_trapezoid(y, dx=1 / sampling_rate_hz, initial=0)

    # bandpass filter
    y = butter_filter(y, sampling_rate_hz, cutoff_freq_hz, f_order, "bandpass")

    if perform_on_flipped_data:
        # flip array to correct direction again
        y = np.flip(y)

    return y


def double_integration_one_side_taper_flipped(
    y: np.ndarray,
    sampling_rate_hz: float,
    tap_s: float,
    cutoff_freq_hz: tuple,
    f_order: int,
) -> np.ndarray:
    """
    Performs a double integration of the flipped, one side tapered signal and filtered signal.

    Flips the original signal to better handle incomplete measurements where the start of train passages are missing.
    If signal has the unit `m/s^2`, then the double integrated signal has the unit `m`.

    Args:
        y (np.ndarray): 1d signal array
        sampling_rate_hz (float): sampling rate of signal array in Hz
        tap_s (float): number of seconds to taper the data array at the beginning and end
        cutoff_freq_hz (tuple): cutoff filter frequencies in Hz (low_hz, high_hz)
        f_order (int): filter order

    Returns:
        np.ndarray: double intergrated signal
    """
    return double_integration(
        y,
        sampling_rate_hz,
        tap_s,
        cutoff_freq_hz,
        f_order,
        start_tapering_only=True,
        perform_on_flipped_data=True,
    )
