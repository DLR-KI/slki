# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
from copy import deepcopy
from typing import Any

import numpy as np
from tqdm.contrib.concurrent import thread_map
from typing_extensions import Self

from ..config import TPrecision
from ..utils.tqdm import get_tqdm_thread_map_kwargs
from .sensor_data import SensorData
from .sensor_data_item import SensorDataItem


TPipelineData = SensorData | list[SensorDataItem]
"""
A type alias representing either a single `SensorData` object or a list of `SensorDataItem` objects.
"""


class SensorDataContainer:
    """
    A container class to hold sensor data and metadata.

    This class provides a simple way to store and manage sensor data,
    along with additional metadata such as resampling status.

    It allows to hold the data as SensorData or as a list of SensorDataItem instances.
    """

    def __init__(self, data: TPipelineData, resampled: bool = False) -> None:
        """
        Initializes a new SensorDataContainer instance.

        Args:
            data (TPipelineData):
                The data to store in this container.
                Can be a single `SensorData` object or a list of `SensorDataItem` objects.
            resampled (bool, optional):  Whether the data has been resampled. Defaults to `False`.
        """
        self.data = data
        """
        The actual sensor data, which can be either a single `SensorData`
        object or a list of `SensorDataItem` objects.
        """
        self.resampled = resampled
        """A flag indicating whether the sensor data has been resampled."""

    @property
    def sample_lengths(self) -> list[int]:
        """
        The sample lengths of each time series.

        Returns:
            List[int]: The sample lengths.
        """
        return self.get_meta("sample_length")

    def get_sensor_data(self) -> SensorData:
        """
        Retrieves the time series data as a `SensorData` object.

        Returns:
            SensorData: The time series data.
        """
        if isinstance(self.data, SensorData):
            return self.data
        return SensorData.from_sensor_data_items(self.data)

    def get_sensor_data_items(self) -> list[SensorDataItem]:
        """
        Retrieves the time series data as a list of `SensorDataItem` objects.

        Returns:
            List[SensorDataItem]: The time series data.
        """
        if isinstance(self.data, SensorData):
            return list(iter(self.data))
        return self.data

    def get_metadata(self) -> list[dict[str, Any]]:
        """
        The meta data for each time series.

        Returns:
            List[Dict[str, Any]]: The meta data.
        """
        if isinstance(self.data, SensorData):
            return self.data.meta
        return [item.meta for item in self.data]

    def get_meta(self, key: str, neutral_element: Any = None) -> list[Any]:
        """
        Retrieves metadata values for each time series based on the given key.

        Args:
            key (str): The metadata key to retrieve.
            neutral_element (Any, optional):
                The value to return if the key is not found in any metadata dictionary. Defaults to `None`.

        Returns:
            List[Any]: A list of metadata values of the given key for all time series.
        """
        if isinstance(self.data, SensorData):
            return self.data.get_meta(key, neutral_element)
        return [item.meta.get(key, neutral_element) for item in self.data]

    def any(self) -> bool:
        """
        Check if the time series data container has any items.

        Returns:
            bool: Whether the sensor data container contains any time series data.
        """
        if isinstance(self.data, SensorData):
            return any(iter(self.data))
        return len(self.data) > 0

    def convert_to_sensor_data(self) -> None:
        """
        Converts the time series data to a single `SensorData` object.

        This method does nothing if the data is already a `SensorData` object.
        For more information about the data convertion, see `SensorData.from_sensor_data_items`.
        """
        self.data = self.get_sensor_data()

    def astype(self, dtype: TPrecision, copy: bool = False) -> Self:
        """
        Casts the time series data of this SensorDataContainer instance to a specified precision.

        Args:
            dtype (TPrecision): The target data type.
            copy (bool, optional): If True, creates a deep copy of the original data before casting. Defaults to False.

        Returns:
            SensorDataContainer: A new or modified SensorDataContainer instance with the casted data.
        """
        me = deepcopy(self) if copy else self
        if isinstance(me.data, SensorData):
            me.data = me.data.astype(dtype)
            return me
        for item in me.data:
            item.data = item.data.astype(dtype)
        return me

    def apply(
        self,
        fn,
        desc: str | None = None,
        tqdm_overwrite_kwargs: dict[str, Any] | None = None,
        *,
        force_iterative: bool = False,
    ) -> None:
        """
        Applies a transformation function to the data.

        This method applies a given function to each item (time series) in the `data` attribute.
        The transformed data is then written back to the original container, updating its internal state.

        If possible the transformation is applied in one step.
        Otherwise given function is called in parallel using multiple threads.

        Args:
            fn (function):
                A callable function that produces the desired data transformation.
                The signature of the function should be:
                `(np.ndarray, List[Dict[str, Any]], List[int], int) -> np.ndarray`.
            desc (Optional[str], optional): Description to display during parallel processing. Defaults to `None`.
            tqdm_overwrite_kwargs (Optional[Dict[str, Any]], optional):
                Additional keyword arguments to pass to `tqdm` for customizing the progress bar. Defaults to `None`.
            force_iterative (bool, optional):
                Whether to force an iterative processing. Defaults to `False`.
        """
        if not isinstance(self.data, SensorData) or force_iterative or not self.resampled:

            def parallel(idx_data_item_tuple: tuple[int, SensorDataItem]):
                idx, data_item = idx_data_item_tuple
                signal, length = data_item.get_signal(self.resampled)
                data_item.data[:length] = fn(
                    data=np.expand_dims(signal, axis=0),
                    metadata=[data_item.meta],
                    lengths=[length],
                    idx=idx,
                ).squeeze(axis=0)

            thread_map(
                parallel,
                list(enumerate(self.data)),
                **get_tqdm_thread_map_kwargs(desc, **(tqdm_overwrite_kwargs or {})),
            )
        else:
            sensor_data = self.data
            sensor_data.data[:] = fn(
                data=sensor_data.data,
                datameta=sensor_data.meta,
                lengths=sensor_data.sample_lengths,
            )

    def __len__(self) -> int:
        """
        Returns the number of time series.

        The number of time series is equal the length of the list of `SensorDataItem` or
        the the number of rows of the data inside the `SensorData` object.

        Returns:
            int: The number of time series.
        """
        return len(self.data)
