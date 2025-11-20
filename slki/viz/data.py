# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""This module provides functions for visualizing sensor data."""

from collections.abc import Sequence
from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np

from ..data.sensor_data import SensorData, SensorDataItem


DATA_DIM_MISMATCH_MSG = "Data dimension mismatch"


def plot_raw_data(sensor_data: SensorData, max_samples: int = 5) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    """
    Plots raw sensor data in separate subplots for each axis.

    Args:
        sensor_data (SensorData): The sensor data to plot.
        max_samples (int, optional): The maximum number of samples to plot. Defaults to 5.

    Raises:
        AssertionError: If the data does not have 3 dimensions (x, y, z).

    Returns:
        Tuple[plt.Figure, plt.Axes]: The matplotlib figure and axes object.
    """
    shape = sensor_data.data.shape
    assert len(shape) == 3 and shape[-1] == 3, "Data must have 3 dimensions (x, y, z)"  # noqa: PLR2004, PT018

    length = min(max_samples, len(sensor_data))
    fig, axes = plt.subplots(length, 3, figsize=(20, 15))
    for signal, axis in zip(sensor_data.get_data(), axes, strict=False):  # type: ignore[arg-type]
        axis[0].plot(signal[:, 0])
        axis[0].set_title("x-axis")
        axis[1].plot(signal[:, 1])
        axis[1].set_title("y-axis")
        axis[2].plot(signal[:, 2])
        axis[2].set_title("z-axis")
    fig.tight_layout()
    return fig, axes


def plot_data(
    sensor_data: SensorData,
    max_samples: int = 5,
    *,
    resampled: bool = True,
) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    """
    Plots sensor data in a single subplot.

    Args:
        sensor_data (SensorData): The sensor data to plot.
        max_samples (int, optional): The maximum number of samples to plot. Defaults to 5.
        resampled (bool, optional): Whether to resample the data. Defaults to True.

    Raises:
        AssertionError: If the data does not have 2 dimensions.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The matplotlib figure and axes object.
    """
    assert len(sensor_data.data.shape) == 2, DATA_DIM_MISMATCH_MSG  # noqa: PLR2004

    length = min(max_samples, len(sensor_data))
    fig, axes = plt.subplots(length, 1, figsize=(20, 15))
    max_sample_length = len(sensor_data.data[0]) if resampled else max(sensor_data.sample_lengths[:max_samples])
    for signal, ax in zip(sensor_data, axes, strict=False):  # type: ignore[arg-type]
        sample_length = max_sample_length if resampled else signal.sample_length
        xs = range(sample_length)
        data = signal.data[:sample_length]
        ax.plot(xs, data)
    fig.tight_layout()
    return fig, axes


def plot_data_comparison(
    raw_data: SensorData,
    processed_data: SensorData,
    max_samples: int = 5,
    *,
    resampled: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plots raw and processed sensor data for comparison.

    Args:
        raw_data (SensorData): The raw sensor data.
        processed_data (SensorData): The processed sensor data.
        max_samples (int, optional): The maximum number of samples to plot. Defaults to 5.
        resampled (bool, optional): Whether to resample the data. Defaults to True.

    Raises:
        AssertionError: If the data does not have 2 dimensions or if the raw and processed data are the same.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The matplotlib figure and axes object.
    """
    assert len(raw_data.data.shape) == 2, DATA_DIM_MISMATCH_MSG  # noqa: PLR2004
    assert len(processed_data.data.shape) == 2, DATA_DIM_MISMATCH_MSG  # noqa: PLR2004
    assert raw_data != processed_data, "Data must be different (at least should not point to the same memory)"

    length = min(max_samples, len(raw_data), len(processed_data))
    fig, axes = plt.subplots(length, 2, figsize=(20, 15))

    max_raw_sample_length = max(raw_data.sample_lengths[:max_samples])
    max_processed_sample_length = (
        len(processed_data.data[0]) if resampled else max(processed_data.sample_lengths[:max_samples])
    )

    for raw_signal, processed_signal, axis in zip(raw_data, processed_data, axes, strict=False):  # type: ignore[arg-type]

        def plot(ax: plt.Axes, signal: SensorDataItem, max_sample_length: int, *, resampled: bool) -> None:
            sample_length = max_sample_length if resampled else signal.sample_length
            xs = range(sample_length)
            data = signal.data[:sample_length]
            ax.plot(xs, data)

        plot(axis[0], raw_signal, max_raw_sample_length, resampled=False)
        plot(axis[1], processed_signal, max_processed_sample_length, resampled=resampled)
    fig.tight_layout()
    return fig, axes


def plot_sample_stages(
    stage_data: Sequence[SensorData],
    stage_titles: Sequence[str] | None = None,
) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    """
    Plots sensor data for different stages in a sequence.

    Args:
        stage_data (Sequence[SensorData]): The sensor data for each stage.
        stage_titles (Optional[Sequence[str]], optional): The titles for each stage. Defaults to None.

    Raises:
        AssertionError: If there is no data to plot or if the number of stage titles is invalid.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The matplotlib figure and axes object.
    """
    assert len(stage_data) > 0, "No data to plot"
    assert stage_titles is None or len(stage_titles) == len(stage_data), "Invalid number of stage titles"
    for stage in stage_data:
        stage.data = stage.data.squeeze()
    assert all(len(stage.data.shape) == 1 for stage in stage_data), DATA_DIM_MISMATCH_MSG

    fig, axes = plt.subplots(len(stage_data), 1, figsize=(20, 15))
    for stage, ax, title in zip_longest(stage_data, axes, stage_titles or []):  # type: ignore[arg-type]
        ax.plot(stage.get_data())
        if title:
            ax.set_title(title)
    fig.tight_layout()
    return fig, axes
