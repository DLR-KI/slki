# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
from collections.abc import Generator, Iterable
from copy import deepcopy
from itertools import chain
from typing import Any

import numpy as np
from tslearn.utils import to_time_series_dataset
from typing_extensions import Self

from ..config import TPrecision
from .sensor_data_item import SensorDataItem


class SensorData:
    """
    Represents a collection of multiple time series.

    This class provides an interface to access and manipulate the data and metadata of multiple time series.
    The data is stored as one large 2D NumPy array, where each row represents a time series.
    Advantage of this representation is that further operations like pre-processing can be done in a efficient way.
    Disadvantage is that it is much more memory consuming since all time series need to have the same length.
    """

    def __init__(self, data: np.ndarray, meta: list[dict[str, Any]]):
        """
        Initializes a SensorData instance.

        Args:
            data (np.ndarray): The time series data as a 2D NumPy array where each row represents a time series.
            meta (List[Dict[str, Any]]): The meta data for each time series.

        Raises:
            AssertionError: If the length of the data array does not match the number of metadata items.
        """
        assert len(data) == len(meta)
        self.data = data
        """The time series data as a 2D NumPy array where each row represents a time series."""
        self.meta = meta
        """The meta data for each time series."""

    @property
    def sample_lengths(self) -> list[int]:
        """
        The sample lengths of each time series.

        Returns:
            List[int]: The sample lengths.
        """
        return self.get_meta("sample_length")

    def get_data(self) -> np.ndarray:
        """
        The time series data as a 2D NumPy array where each row represents a time series.

        Returns:
            np.ndarray: The time series data.
        """
        return self.data

    def get_metadata(self) -> list[dict[str, Any]]:
        """
        The meta data for each time series.

        Returns:
            List[Dict[str, Any]]: The meta data.
        """
        return self.meta

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
        return [m.get(key, neutral_element) for m in self.meta]

    def extend(self, *others: Self) -> None:
        """
        Extends this SensorData instance by combining it with one or more other SensorData instances.

        The data and metadata of all combined SensorData instances are merged into a single instance.
        Please note that all data arrays must have the same time series length.

        Raises:
            AssertionError: If the underlying data arrays do not have the same time series data length.

        Args:
            *others: Variable number of other SensorData instances to combine with this instance.
        """
        combined = self.__class__.concatenate(self, *others)
        self.data = combined.data
        self.meta = combined.meta

    def copy(self) -> Self:
        """
        Creates a deep copy of this SensorData instance.

        Returns:
            SensorData: A new SensorData instance with the same data and metadata as this instance.
        """
        return deepcopy(self)

    def astype(self, dtype: TPrecision, copy: bool = False) -> Self:
        """
        Casts the time series data of this SensorData instance to a specified precision.

        Args:
            dtype (TPrecision): The target data type.
            copy (bool, optional): If True, creates a deep copy of the original data before casting. Defaults to False.

        Returns:
            SensorData: A new or modified SensorData instance with the casted data.
        """
        me = self.copy() if copy else self
        me.data = me.data.astype(dtype)
        return me

    def __len__(self) -> int:
        """
        Returns the number of time series in this SensorData instance.

        This length corresponds to the number of rows in the underlying data array.

        Returns:
            int: The number of time series.
        """
        return len(self.data)

    def __iter__(self) -> Generator[SensorDataItem, None, None]:
        """
        Returns an iterator over the time series data in this SensorData instance.

        Each iteration yields a SensorDataItem object containing a single time series and its associated metadata.

        Yields:
            SensorDataItem: The next data row or time series and its associated metadata.
        """
        for data, meta in zip(self.data, self.meta, strict=False):
            yield SensorDataItem(data, meta)

    @classmethod
    def from_sensor_data_items(cls, data_items: Iterable[SensorDataItem]) -> Self:
        """
        Creates a SensorData instance from an iterable of SensorDataItem objects.

        This method extracts the time series data and metadata from each SensorDataItem and combines them into a
        new SensorData instance. It is important to know that all time series must have the same length to be able
        to create a SensorData instance due to the underlying 2D NumPy array representation. Therefore, the data
        will be automatically transformed. In general, that means that all time series will be extended with zeros
        to have the same length as the longest time series. This is done by the `tslearn.utils.to_time_series_dataset`.
        Note that this increases the memory consumption.

        Args:
            data_items (Iterable[SensorDataItem]):
                An iterable of SensorDataItem objects containing the time series and metadata.

        Returns:
            SensorData: A new SensorData instance constructed from the provided data items.
        """
        data, meta = zip(*data_items, strict=False)
        np_data = to_time_series_dataset(data)
        np_data = np_data.squeeze(axis=-1)
        return cls(np_data, list(meta))

    @classmethod
    def concatenate(cls, *sensor_data: Self) -> Self:
        """
        Concatenates multiple SensorData instances into a single instance.

        The time series and metadata from each input instance are combined.
        Please note that all data arrays must have the same time series length.

        Args:
            *sensor_data: Variable number of SensorData instances to concatenate.

        Raises:
            AssertionError: If the underlying data arrays do not have the same time series data length.

        Returns:
            SensorData: A new SensorData instance containing the combined time series data as well as metadata.
        """
        assert len(sensor_data) > 0
        if len(sensor_data) == 1:
            return sensor_data[0]

        first, *others = sensor_data
        assert all(sensor_data.data.shape[-1] == first.data.shape[-1] for sensor_data in others), (
            "all sensor data must have the same length (consider resampling are padding)"
        )

        return cls(
            np.vstack([sensor_data.data for sensor_data in sensor_data]),
            list(chain(*(sensor_data.meta for sensor_data in sensor_data))),
        )
