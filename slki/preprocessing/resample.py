# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
import numpy as np
from tslearn.preprocessing import TimeSeriesResampler
from typing_extensions import override

from ..config import Config
from ..data.sensor_data import SensorData
from ..data.sensor_data_item import SensorDataItem
from .stage import Stage


class Resample(Stage):
    """A preprocessing stage that resamples one or more time series to a specified frequency."""

    SUPPORTS_ONLY_ITERATIVE: bool = True

    def run(self, resample_size: int = 10000, *, force: bool = False) -> None:
        """
        Runs the resampling stage.

        Resamples the input data to the specified size. If the data has already been resampled and `force` is False,
        a warning message is logged and the method returns without further processing.

        Args:
            resample_size (int, optional): The desired size of the resampled time series. Defaults to 10000.
            force (bool, optional): If True, forces resampling even if it has already been done. Defaults to False.
        """
        # check if the data has already been resampled
        if self.data_container.resampled:
            mode = "requested" if force else "undesired"
            msg = f"Data has already been resampled. Force resampling is {mode}."
            if force:
                self.logger.warning(msg + " Resample again.")
            else:
                self.logger.warning(msg + " Skip resampling.")
                return

        if isinstance(self.data_container.data, SensorData):
            sensor_data = self.data_container.data
            sensor_data.data = self._resample_np_matrix(sensor_data.data, sensor_data.sample_lengths, resample_size)
        else:
            sensor_data_items = self.data_container.data
            self._resample_sensor_data_items(sensor_data_items, resample_size)

        # remember that the data has been resampled
        self.data_container.resampled = True

        # boost further processing stages by converting all data into a single large matrix
        self.data_container.convert_to_sensor_data()

    @override
    def run_default(self) -> None:
        self.run(Config.RESAMPLE_SIZE)

    def _resample_np_matrix(self, data: np.ndarray, lengths: list[int], resample_size: int) -> np.ndarray:
        """
        Resamples a 2D NumPy matrix of time series to a specified frequency.

        This method extends or truncates the input data as needed, and applies the time series resampling.

        Args:
            data (np.ndarray): The input 2D NumPy array containing multiple time series.
            lengths (List[int]): A list of desired sample lengths for each time series in `data`.
            resample_size (int): The target size of the resampled time series.

        Returns:
            np.ndarray: The resampled 2D NumPy matrix.
        """
        # extend the time series if necessary
        original_length = len(data[0])
        if resample_size > original_length:
            data = np.pad(
                data,
                [(0, 0), (0, resample_size - original_length)],
                mode="constant",
                constant_values=0,
            )

        # resample
        for signal, length in zip(data, lengths, strict=False):
            real_signal_length = len(signal) if self.data_container.resampled else length
            if real_signal_length < 10:
                self.logger.warning('"real" signal length smaller than 10. Skip resampling.')
                continue
            s = signal[:real_signal_length]
            signal[:resample_size] = self._resample(s, resample_size)

        # ensure that all time series have the desired length
        return data[:, :resample_size]

    def _resample_sensor_data_items(self, sensor_data_items: list[SensorDataItem], resample_size: int) -> None:
        """
        Resamples a list of SensorDataItem instances.

        This method iterates over the input data, applies the time series resampling for each item, and updates
        the corresponding SensorDataItem instances.

        Args:
            sensor_data_items (List[SensorDataItem]): The list of SensorDataItem instances to resample.
            resample_size (int): The target size of the resampled time series.
        """
        for data_item in sensor_data_items:
            length = len(data_item.data) if self.data_container.resampled else data_item.sample_length
            signal = data_item.data[:length]
            data_item.data = self._resample(signal, resample_size)

    def _resample(self, signal: np.ndarray, resample_size: int) -> np.ndarray:
        """
        Resamples a single 1D NumPy array representing a time series.

        This method uses a TimeSeriesResampler instance to apply the resampling operation.

        Args:
            signal (np.ndarray): The input 1D NumPy array containing a time series.
            resample_size (int): The target size of the resampled time series.

        Returns:
            np.ndarray: The resampled 1D NumPy array.
        """
        resampler = TimeSeriesResampler(resample_size)
        return resampler.fit_transform(signal).flatten()
