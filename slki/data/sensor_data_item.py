# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
from datetime import datetime, timezone
from typing import Any

import numpy as np


class SensorDataItem:
    """
    Represents a single data item from a sensor, containing its 1D time series data and associated metadata.

    This class provides an interface to access and manipulate the data and metadata of a single data point.
    """

    def __init__(self, data: np.ndarray, meta: dict[str, Any]):
        """
        Initializes a SensorDataItem instance.

        Args:
            data (np.ndarray): The sensor 1D time series data.
            meta (Dict[str, Any]):
                Meta data about the sensor data including at least `"sample_length"` of the data.
        """
        self.data = data
        """The sensor data as 1D time series."""
        self.meta = meta
        """The meta data."""

    @property
    def sample_length(self) -> int:
        """
        The sample length of the time series data.

        Returns:
            int: The sample length.
        """
        return self.meta["sample_length"]

    @sample_length.setter
    def sample_length(self, value: int):
        """
        Sets the sample length for this data item.

        Args:
            value (int): The new sample length.
        """
        self.meta["sample_length"] = value

    def get_signal(self, resampled: bool) -> tuple[np.ndarray, int]:
        """
        Get the signal data with the correct sample length and its sample length.

        Args:
            resampled (bool):
                Wether the signal is already resampled of not.
                If True, returns the entire data array.
                Otherwise, returns a slice of the data up to its sample length.

        Returns:
            Tuple[np.ndarray, int]: A tuple containing the signal data and its length.
        """
        length = len(self.data) if resampled else self.sample_length
        return self.data[:length], length

    def __iter__(self):
        """
        Returns an iterator over this SensorDataItem instance.

        Yields:
            The data array and metadata dictionary, in that order.

        Example:
            ```python
            data, meta = sensor_data_item
            ```
        """
        yield self.data
        yield self.meta

    @classmethod
    def create(
        cls,
        data: np.ndarray,
        start_time: str | datetime | None = None,
        end_time: str | datetime | None = None,
        *,
        sample_length: int | None = None,
        **meta_kwargs,
    ) -> "SensorDataItem":
        """
        Creates a new SensorDataItem instance from the given data and metadata.

        Args:
            data (np.ndarray): The sensor data (time series data).
            start_time (str | datetime | None, optional):
                The start time of the data as a string or datetime object, or `None` for unknown start time.
                Defaults to `None`.
            end_time (str | datetime | None, optional):
                The end time of the data as a string or datetime object, or `None` for unknown end time.
                Defaults to `None`.
            sample_length (int | None, optional):
                The number of samples in the data array.
                If not provided, it will be inferred from the data length.
                Defaults to None.
            **meta_kwargs: Additional metadata key-value pairs.

        Returns:
            SensorDataItem: A new SensorDataItem instance with the given data and metadata.
        """
        meta: dict[str, Any] = {}
        meta["sample_length"] = sample_length if sample_length is not None else len(data)
        if start_time is not None and end_time is not None:
            meta["start_time"] = cls._parse_timestamp(start_time)
            meta["end_time"] = cls._parse_timestamp(end_time)
        meta.update(meta_kwargs)
        return cls(data, meta)

    @classmethod
    def _parse_timestamp(cls, timestamp: str | datetime | None) -> datetime | None:
        """
        Parses a timestamp from a string or datetime object into a datetime object.

        Args:
            timestamp (str | datetime | None):
                The timestamp to parse as a string or datetime object, or `None` for unknown start time.
                The timestamp string should be in ISO format with a trailing 'Z' (e.g., '2024-02-16T13:23:00.657810Z').

        Returns:
            datetime | None: The parsed datetime object, or `None` if the input was `None`.
        """
        if timestamp is None:
            return None
        if isinstance(timestamp, datetime):
            return timestamp
        # timestamps inside hdf5 file are in iso format with a tailing 'Z': '2024-02-16T13:23:00.657810Z'
        # fromisoformat reads the string without the 'Z' at the end
        return datetime.fromisoformat(timestamp[:-1]).astimezone(timezone.utc)
