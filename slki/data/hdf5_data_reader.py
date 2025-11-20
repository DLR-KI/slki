# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from itertools import islice
import logging
import re
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

from ..config import Config, TAxis, TSensor, TSensorData
from ..utils.tqdm import get_tqdm_kwargs
from .sensor_data_item import SensorDataItem


@dataclass
class Origin:
    """Data origin to simplify logging and signal tracking."""

    file: str
    """The path to the HDF5 file containing the dataset."""
    dataset: str
    """The name/path of the HDF5 dataset within the file."""

    def serialize(self) -> dict[str, str]:
        """
        Returns a dictionary representation of the Origin instance.

        Returns:
            Dict[str, str]: A dictionary representation of the Origin instance.
        """
        return dict(file=self.file, dataset=self.dataset)

    def __str__(self) -> str:
        """
        Returns a minimalistic single line representation.

        Returns:
            str: Returns a string representation of the Origin instance.
        """
        return f"Origin[file={self.file},dataset={self.dataset}]"


@dataclass
class LogInfo:
    """Simple helper class containing log-related information."""

    origin: Origin
    """The origin of the dataset being read."""
    logger: logging.Logger | None = logging.root
    """The logger to use for logging events. Defaults to the root logger."""


def build_default_dataset_name_pattern(sensor_name: TSensorData = Config.SENSOR_DATA) -> str:
    """
    Builds the default dataset name pattern for an HDF5 file.

    Args:
        sensor_name (TSensorData, optional): The name of the sensor. Defaults to `Config.SENSOR_DATA`.

    Returns:
        str: The default dataset name pattern.
    """
    return f"/su_acceleration_data/{sensor_name}$"


def get_sensor_datasets(
    group: h5py.Group,
    name_pattern: str | re.Pattern[str] | None = build_default_dataset_name_pattern(),  # noqa: B008
    name_regex_flags: re.RegexFlag = 0,  # type: ignore[assignment]  # 0 = re.RegexFlag.NOFLAG
) -> Generator[h5py.Dataset, None, None]:
    """
    Find HDF5 dataset objects that match the specified name pattern.

    Args:
        group (h5py.Group): The top-level HDF5 group to search.
        name_pattern (Optional[str | re.Pattern[str]], optional):
            A string or regular expression pattern to match dataset names.
            Defaults to pattern built by `build_default_dataset_name_pattern()`.
        name_regex_flags (re.RegexFlag, optional):
            Regular expression flags to use when matching patterns. Defaults to no flags.

    Yields:
        h5py.Dataset: The next HDF5 dataset that match the specified pattern.
    """
    for item in group.values():
        if isinstance(item, h5py.Group):
            yield from get_sensor_datasets(item, name_pattern, name_regex_flags)
        elif name_pattern is None or re.search(name_pattern, item.name, name_regex_flags):
            yield item  # type: ignore[misc]


def get_sensor_dataset_names(
    group: h5py.Group,
    name_pattern: str | re.Pattern[str] | None = build_default_dataset_name_pattern(),  # noqa: B008
    name_regex_flags: re.RegexFlag = 0,  # type: ignore[assignment]  # 0 = re.RegexFlag.NOFLAG
) -> Generator[str, None, None]:
    """
    Find the names of HDF5 datasets that match the specified name pattern.

    Args:
        group (h5py.Group): The top-level HDF5 group to search.
        name_pattern (Optional[str | re.Pattern[str]], optional):
            A string or regular expression pattern to match dataset names.
            Defaults to pattern built by `build_default_dataset_name_pattern()`.
        name_regex_flags (re.RegexFlag, optional):
            Regular expression flags to use when matching patterns. Defaults to no flags.

    Yields:
        str: The next HDF5 dataset name that match the specified pattern.
    """
    for ds in get_sensor_datasets(group, name_pattern, name_regex_flags):
        yield ds.name


def get_attr_or_log_warning(
    obj: h5py.Group | h5py.Dataset | h5py.Datatype,
    attr_name: str,
    log_info: LogInfo,
    skip_msg: bool = False,
) -> Any:
    """
    Retrieves an HDF5 attribute by name, or logs a warning if it is not found.

    Args:
        obj (h5py.Group | h5py.Dataset | h5py.Datatype): The HDF5 object to search for the attribute.
        attr_name (str): The name of the attribute to retrieve.
        log_info (LogInfo): Information about the logging context.
        skip_msg (bool, optional):
            Whether to skip the warning message if the attribute is not found. Defaults to False.

    Returns:
        Any: The value of the attribute if it exists, or None if it does not exist.
    """
    if obj.attrs.__contains__(attr_name):
        return obj.attrs[attr_name]
    if log_info.logger:
        msg = f"No '{attr_name}' attribute found in '{log_info.origin}."
        if skip_msg:
            msg += " Skip."
        log_info.logger.warning(msg)
    return None


def extract_axis(data: np.ndarray, axis: TAxis = Config.AXIS) -> np.ndarray:
    """
    Extracts a specified axis from a 2D NumPy array.

    Args:
        data (np.ndarray): The input array, expected to be Nx3.
        axis (TAxis, optional): The axis to extract. Defaults to Config.AXIS.

    Raises:
        ValueError: If an invalid axis is specified.

    Returns:
        np.ndarray: A 1D NumPy array containing the extracted data.
    """
    assert data.ndim == 2 and data.shape[-1] == 3, "Data shape missmatch."
    if axis == "x":
        return data[:, 0]
    if axis == "y":
        return data[:, 1]
    if axis == "z":
        return data[:, 2]
    if axis == "a":
        return np.mean(data, axis=1)
    raise ValueError(f"Invalid axis: '{axis}'")


def value_in_boundary(
    value: float,
    lower_bound: float | None,
    upper_bound: float | None,
    desc: str,
    log_info: LogInfo,
) -> bool:
    """
    Checks if a value is within the specified bounds, and logs a warning otherwise.

    Args:
        value (int | float): The value to check.
        lower_bound (Optional[int | float]): The lower bound. `None` for no lower bound.
        upper_bound (Optional[int | float]): The upper bound. `None` for no upper bound.
        desc (str): A description of the object being checked, used in the log message.
        log_info (LogInfo): Information about the logging context.

    Returns:
        bool: True if the value is within bounds, False otherwise.
    """
    lower_bound = lower_bound or float("-inf")
    upper_bound = upper_bound or float("inf")
    if lower_bound <= value <= upper_bound:
        return True
    if log_info.logger:
        log_info.logger.warning(
            f"{desc} '{value}' out of bounds [{lower_bound}, {upper_bound}] for {log_info.origin}. Skip.",
        )
    return False


class HDF5DataReader:
    """A data reader for HDF5 files."""

    SUPPORTED_FILE_EXTENSIONS = [".hdf", ".h4", ".hdf4", ".he2", ".h5", ".hdf5", ".he5"]
    """The file extensions supported by this reader."""
    SUPPORTED_FILE_MINETYPES = ["application/x-hdf", "application/x-hdf5"]
    """The MIME types of files supported by this reader."""

    def __init__(
        self,
        sensor_type: TSensor = Config.SENSOR,
        sensor_name: TSensorData = Config.SENSOR_DATA,
        metadata_only: bool = Config.LOAD_METADATA_ONLY,
        axis: TAxis = Config.AXIS,
        sample_length_lower_bound: int | None = Config.SAMPLE_LENGTH_LOWER_BOUND,
        sample_length_upper_bound: int | None = Config.SAMPLE_LENGTH_UPPER_BOUND,
        sample_rate_lower_bound: int | None = Config.SAMPLE_RATE_LOWER_BOUND,
        sample_rate_upper_bound: int | None = Config.SAMPLE_RATE_UPPER_BOUND,
        tqdm_options: dict[str, Any] = Config.TQDM_OPTIONS,
        root_path: str = "/su_data",
        logger_name: str = "slki.data.hdf5_data_reader",
    ):
        """
        Initializes the HDF5DataReader with various configuration options.

        Args:
            sensor_type (TSensor, optional): The type of sensor data to read. Defaults to `Config.SENSOR`.
            sensor_name (TSensorData, optional): The name of the sensor data to read. Defaults to `Config.SENSOR_DATA`.
            metadata_only (bool, optional): Whether to load only metadata or also data. Defaults to `Config.LOAD_METADATA_ONLY`.
            axis (TAxis, optional): The axis to extract from the data. Defaults to `Config.AXIS`.
            sample_length_lower_bound (Optional[int], optional): The lower bound of sample length. Defaults to `Config.SAMPLE_LENGTH_LOWER_BOUND`.
            sample_length_upper_bound (Optional[int], optional): The upper bound of sample length. Defaults to `Config.SAMPLE_LENGTH_UPPER_BOUND`.
            sample_rate_lower_bound (Optional[int], optional): The lower bound of sample rate. Defaults to `Config.SAMPLE_RATE_LOWER_BOUND`.
            sample_rate_upper_bound (Optional[int], optional): The upper bound of sample rate. Defaults to `Config.SAMPLE_RATE_UPPER_BOUND`.
            tqdm_options (Dict[str, Any], optional): Options for the tqdm progress bar. Defaults to `Config.TQDM_OPTIONS`.
            root_path (str, optional): The root path where data is stored. Defaults to `"/su_data"`.
            logger_name (str, optional): The name of the logger to use. Defaults to `"slki.data.hdf5_data_reader"`.
        """  # noqa: E501
        self.sensor_type = sensor_type
        """The type of sensor data read."""
        self.sensor_name = sensor_name
        """The name of the sensor data read."""
        self.metadata_only = metadata_only
        """Whether only metadata or also data was loaded."""
        self.axis = axis
        """The axis extracted from the data."""
        self.sample_length_lower_bound = sample_length_lower_bound
        """The lower bound of sample length."""
        self.sample_length_upper_bound = sample_length_upper_bound
        """The upper bound of sample length."""
        self.sample_rate_lower_bound = sample_rate_lower_bound
        """The lower bound of sample rate."""
        self.sample_rate_upper_bound = sample_rate_upper_bound
        """The upper bound of sample rate."""
        self.tqdm_options = tqdm_options
        """Options for the tqdm progress bar."""
        self.root_path = root_path
        """The root path where data is stored."""
        self.logger = logging.getLogger(logger_name)
        """The logger instance."""

    def load_all(
        self,
        filepaths: Sequence[str],
        max_samples: int | None = Config.MAX_SAMPLES_PER_SENDOR_TYPE,
    ) -> list[SensorDataItem]:
        """
        Loads the sensor data items from the specified files.

        Args:
            filepaths (Sequence[str]): A sequence of paths to HDF5 files.
            max_samples (Optional[int], optional):
                The maximum number of samples to load.
                `None` to load all. Defaults to `Config.MAX_SAMPLES_PER_SENDOR_TYPE`.

        Returns:
            List[SensorDataItem]: A list of loaded sensor data items.
        """
        data_generator, _ = self.load(filepaths, disable_data_loading_progress=False)
        return list(islice(data_generator, max_samples))

    def load(
        self,
        filepaths: Sequence[str],
        disable_data_loading_progress: bool = True,
    ) -> tuple[Generator[SensorDataItem, None, None], int]:
        """
        Loads data from the specified HDF5 files.

        Args:
            filepaths (Sequence[str]): A sequence of paths to HDF5 files.
            disable_data_loading_progress (bool, optional):
                Whether to show or hide the progress bar. Defaults to True.

        Returns:
            Tuple[Generator[SensorDataItem, None, None], int]:
                A tuple containing a generator of sensor data items and the estimated maximum signal size.
                The maximum signal size is only an estimate and can be less than the actual number of valid
                signals due to skipping invalid data during loading.
        """
        # analyse data
        sensor_dataset_names_per_hdf5 = self._get_dataset_path(filepaths)
        max_signal_size = sum(
            len(sensor_dataset_names) for sensor_dataset_names in sensor_dataset_names_per_hdf5.values()
        )
        self.logger.info(
            f"Found {max_signal_size} potential signals in {len(sensor_dataset_names_per_hdf5)} HDF5 files.",
        )

        # define data generator
        data_info_iterator = tqdm(
            iterable=((k, v) for k, values in sensor_dataset_names_per_hdf5.items() for v in values),
            total=sum(len(sensor_dataset_names) for sensor_dataset_names in sensor_dataset_names_per_hdf5.values()),
            **get_tqdm_kwargs(desc="Loading HDF5 data", enable=not disable_data_loading_progress),
        )
        data_generator = self._load_data(data_info_iterator)

        # return data generator and max signal size
        # max signal size is only an estimation (it can be less, but not more),
        # since invalid data will automatically be skipped
        return data_generator, max_signal_size

    def _load_data(self, data_info_iterator: Iterable[tuple[str, str]]) -> Generator[SensorDataItem, None, None]:
        """
        Loads and yields SensorDataItem instances from the specified HDF5 files.

        Args:
            data_info_iterator (Iterable[Tuple[str, str]]):
                An iterator yielding tuples of (filepath, sensor_dataset_name).

        Yields:
            SensorDataItem: The next loaded sensor data item.
        """
        for filepath, sensor_dataset_name in data_info_iterator:
            with h5py.File(filepath, "r") as hdf:  # type: ignore[arg-type, call-arg]
                ds: h5py.Dataset = hdf[sensor_dataset_name]  # type: ignore[assignment]
                log_info = LogInfo(Origin(filepath, sensor_dataset_name), logger=self.logger)

                data = self._get_data(ds, log_info)
                if data is None:
                    continue

                metadata = self._get_metadata(ds, data, log_info)
                if metadata is None:
                    continue

            data_item = SensorDataItem.create(data, **metadata)

            # validate ratio of sample rate and sample length to start and end time stamps
            delta_timestamp: timedelta = data_item.meta["end_time"] - data_item.meta["start_time"]
            delta_sample_rate = timedelta(seconds=data_item.sample_length / data_item.meta["sample_rate_in_hz"])
            diff = abs(delta_timestamp - delta_sample_rate)
            if diff.total_seconds() > 0.01:
                # delta difference is larger than 10 milliseconds
                action = "Skip!" if Config.SKIP_TIMESTAMP_MISSMATCH else "Still take them!"
                self.logger.warning(
                    f"Sample length and sample rate does not match time span (end - start). {action}"
                    f" -- abs( {delta_sample_rate} - {delta_timestamp} ) = {diff} > 10ms -- [{log_info.origin}]",
                )
                if Config.SKIP_TIMESTAMP_MISSMATCH:
                    continue

            yield data_item

    def _get_dataset_path(self, filepaths: Sequence[str]) -> dict[str, list[str]]:
        """
        Fetching all relevant sensor dataset names mapped to the specified HDF5 files.

        Args:
            filepaths (Sequence[str]): A sequence of paths to HDF5 files.

        Returns:
            Dict[str, List[str]]:
                A dictionary where each key is an HDF5 file path and each value is a list of
                corresponding sensor dataset names.
        """
        sensor_dataset_names: dict[str, list[str]] = {}
        for filepath in tqdm(filepaths, **get_tqdm_kwargs(desc="Analysing HDF5 files", enable=len(filepaths) > 1)):
            self.logger.debug(f"Analysing HDF5 file: '{filepath}'")

            try:
                with h5py.File(filepath, "r") as hdf:  # type: ignore[arg-type, call-arg]
                    if not hdf.__contains__(self.root_path):
                        self.logger.warning(f"HDF5 file does not contain '{self.root_path}'. Skipping file: {filepath}")
                        continue
                    sensor_dataset_names[filepath] = []

                    sensor: h5py.Group
                    for sensor in tqdm(
                        hdf[self.root_path].values(),  # type: ignore[union-attr]
                        **get_tqdm_kwargs(desc="Sensors", enable=self.sensor_type == "both"),
                    ):
                        if self.sensor_type != "both" and self.sensor_type != str(sensor.attrs["placement"]):
                            # sensor attribute "placement" can be "points" or "frog"
                            continue  # skip unwanted sensor types

                        # fetch all relevant hdf5 file dataset paths
                        sensor_dataset_names[filepath].extend(
                            get_sensor_dataset_names(sensor, build_default_dataset_name_pattern(self.sensor_name)),
                        )
            except OSError as e:
                self.logger.error(str(e) + f" => Skipping file '{filepath}'")
        return sensor_dataset_names

    def _get_data(self, ds: h5py.Dataset, log_info: LogInfo) -> np.ndarray | None:
        """
        Loads data from the specified HDF5 dataset, returning the numpy array representation of the data if valid.

        This methods loads the data from the HDF5 dataset in its numpy array representation.
        It will also extract the desired axis and transform the data from counts to accelaration in g (m/s**2).

        Args:
            ds (h5py.Dataset): The HDF5 dataset to load.
            log_info (LogInfo): Logging information for error messages.

        Returns:
            Optional[np.ndarray]:
                A 1D numpy array containing the loaded data or `None` if the data was invalid.
                If metadata-only mode is enabled, this method will be skip and an empty 1D numpy array is returned.
        """
        if self.metadata_only:
            return np.array([])

        # counts_to_g_transform represent the acc data multiplying factor
        counts_to_g_transform = get_attr_or_log_warning(ds, "counts_to_g_transform", log_info, skip_msg=True)
        if counts_to_g_transform is None:
            return None

        # ds[:] returns the numpy data representation
        data = ds[:]
        # transform counts to accelaration in g
        data = data * counts_to_g_transform
        # transform accelaration in g to accelaration in m/s**2 (g * 9.81 = m/s**2)
        data = data * 9.81
        # extract the desired axis
        data = extract_axis(data, self.axis)

        # validate data
        if np.isnan(data).any():
            self.logger.warning(f"Invalid data (containing NaN) found in '{ds.name}'. Skip.")
            return None
        if np.isinf(data).any():
            self.logger.warning(f"Invalid data (containing infinity) found in '{ds.name}'. Skip.")
            return None

        return data

    def _get_metadata(self, ds: h5py.Dataset, data: np.ndarray, log_info: LogInfo) -> dict[str, Any] | None:
        """
        Loads and validates metadata from the specified HDF5 dataset.

        Args:
            ds (h5py.Dataset): The HDF5 dataset to load.
            data (np.ndarray): The loaded data array (used for sample length validation).
            log_info (LogInfo): Logging information for error messages.

        Returns:
            Optional[Dict[str, Any]]:
                A dictionary containing the loaded metadata, or `None` if any validation failed.
                The returned metadata dictionary is ready to be passed to `SensorDataItem.create(data, **metadata)`.
        """
        # fetch metadata
        sample_length = len(data) if data.any() else ds.shape[0]
        start_time = get_attr_or_log_warning(ds, "seq_start", log_info)
        end_time = get_attr_or_log_warning(ds, "seq_end", log_info)
        sample_rate_in_hz = get_attr_or_log_warning(ds, "sample_rate_in_hz", log_info, skip_msg=True)
        if sample_rate_in_hz is None:
            return None

        # validate metadata
        if not value_in_boundary(
            sample_length,
            self.sample_length_lower_bound or -np.inf,
            self.sample_length_upper_bound or np.inf,
            "Sample length",
            log_info,
        ):
            return None
        if not value_in_boundary(
            sample_rate_in_hz,
            self.sample_rate_lower_bound or -np.inf,
            self.sample_rate_upper_bound or np.inf,
            "Sample rate",
            log_info,
        ):
            return None

        # ready for `SensorDataItem.create(data, **metadata)`
        return dict(
            start_time=start_time,
            end_time=end_time,
            sample_length=sample_length,
            sample_rate_in_hz=sample_rate_in_hz,
            origin=log_info.origin.serialize(),
        )
