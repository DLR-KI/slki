# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
from collections.abc import Generator
from itertools import zip_longest
from logging import getLogger
import os
import pickle
import re
from typing import Any, overload

import numpy as np

from ..config import Config, TNorm
from ..preprocessing import Normalize, Resample
from ..utils.path import are_files_readable
from .sensor_data import SensorData
from .sensor_data_container import SensorDataContainer


_IMPORT_LOGGER_NAME = "slki.data.import"
"""Logger name for the data import module."""


@overload
def load_all_typed(
    *,
    resample_size: int | None = None,
    normalize: TNorm | None = None,
) -> SensorData:
    """
    Loads all data from the specified files and combines them into a single `SensorData` object.

    This function builds the data and meta data paths based on the configuration.
    It automatically detects whether the data is split into multiple chunks, loads each
    chunk of data separately, and then concatenates these chunks together to form a complete
    `SensorData` object. If only a single chunk is found, that chunk is returned directly.

    Args:
        resample_size (Optional[int], optional):
            Size to which the loaded time series should be resampled.
            If `None`, no resampling is performed. Defaults to `None`.
        normalize (Optional[TNorm], optional):
            Normalization method to apply to the loaded time series.
            If `None`, no normalization is performed. Defaults to `None`.

    Returns:
        SensorData:
            A `SensorData` object containing all loaded data, either directly if only one chunk
            exists or after concatenating multiple chunks together.
    """


@overload
def load_all_typed(
    datapath: str,
    *,
    resample_size: int | None = None,
    normalize: TNorm | None = None,
) -> SensorData:
    """
    Loads all data from the specified files and combines them into a single `SensorData` object.

    This function takes in data file path and generates the meta data file path based on this data file path.
    It automatically detects whether the data is split into multiple chunks, loads each
    chunk of data separately, and then concatenates these chunks together to form a complete
    `SensorData` object. If only a single chunk is found, that chunk is returned directly.

    Note, there exists special placeholder for the chunk number in the file paths:

    - `%(chunk)` for any series of digits representable by the Regex pattern `[0-9]+` and
    - `%(chunk?)` as an optional chunking placeholder representable by the Regex pattern `(_?[0-9]+)?`.

    Args:
        datapath (str): The path(s) where the data file(s) are located.
        resample_size (Optional[int], optional):
            Size to which the loaded time series should be resampled.
            If `None`, no resampling is performed. Defaults to `None`.
        normalize (Optional[TNorm], optional):
            Normalization method to apply to the loaded time series.
            If `None`, no normalization is performed. Defaults to `None`.

    Returns:
        SensorData:
            A `SensorData` object containing all loaded data, either directly if only one chunk
            exists or after concatenating multiple chunks together.
    """


@overload
def load_all_typed(
    datapath: str,
    metapath: str,
    *,
    resample_size: int | None = None,
    normalize: TNorm | None = None,
) -> SensorData:
    """
    Loads all data from the specified files and combines them into a single `SensorData` object.

    This function takes in the paths to data and metadata files.
    It automatically detects whether the data is split into multiple chunks, loads each
    chunk of data separately, and then concatenates these chunks together to form a complete
    `SensorData` object. If only a single chunk is found, that chunk is returned directly.

    Note, there exists special placeholder for the chunk number in the file paths:

    - `%(chunk)` for any series of digits representable by the Regex pattern `[0-9]+` and
    - `%(chunk?)` as an optional chunking placeholder representable by the Regex pattern `(_?[0-9]+)?`.

    Args:
        datapath (str): The path(s) where the data file(s) are located.
        metapath (str): The path(s) where the metadata file(s) are located.
        resample_size (Optional[int], optional):
            Size to which the loaded time series should be resampled.
            If `None`, no resampling is performed. Defaults to `None`.
        normalize (Optional[TNorm], optional):
            Normalization method to apply to the loaded time series.
            If `None`, no normalization is performed. Defaults to `None`.

    Returns:
        SensorData:
            A `SensorData` object containing all loaded data, either directly if only one chunk
            exists or after concatenating multiple chunks together.
    """


def load_all_typed(
    datapath: str | None = None,
    metapath: str | None = None,
    *,
    resample_size: int | None = None,
    normalize: TNorm | None = None,
) -> SensorData:
    """
    Loads all data from the specified files and combines them into a single `SensorData` object.

    This function takes in one or both of the paths to data and metadata files.
    It automatically detects whether the data is split into multiple chunks, loads each
    chunk of data separately, and then concatenates these chunks together to form a complete
    `SensorData` object. If only a single chunk is found, that chunk is returned directly.

    Note, there exists special placeholder for the chunk number in the file paths:

    - `%(chunk)` for any series of digits representable by the Regex pattern `[0-9]+` and
    - `%(chunk?)` as an optional chunking placeholder representable by the Regex pattern `(_?[0-9]+)?`.

    Args:
        datapath (Optional[str], optional):
            The path(s) where the data file(s) are located.
            If `None`, the path will be build from the configuration. Defaults to `None`.
        metapath (Optional[str], optional):
            The path(s) where the metadata file(s) are located.
            If `None`, the path will be build based on the `datapath`. Defaults to `None`.
        resample_size (Optional[int], optional):
            Size to which the loaded time series should be resampled.
            If `None`, no resampling is performed. Defaults to `None`.
        normalize (Optional[TNorm], optional):
            Normalization method to apply to the loaded time series.
            If `None`, no normalization is performed. Defaults to `None`.

    Returns:
        SensorData:
            A `SensorData` object containing all loaded data, either directly if only one chunk
            exists or after concatenating multiple chunks together.
    """
    gen = load_typed(  # type: ignore[misc]
        datapath,  # type: ignore[arg-type]
        metapath,  # type: ignore[arg-type]
        resample_size=resample_size,
        normalize=normalize,
    )
    data_chunks = list(gen)
    if len(data_chunks) == 1:
        return data_chunks[0]
    getLogger(_IMPORT_LOGGER_NAME).info("Concatenate all data chunks.")
    return SensorData.concatenate(*data_chunks)


@overload
def load_typed(
    *,
    resample_size: int | None = None,
    normalize: TNorm | None = None,
) -> Generator[SensorData, None, None]:
    """
    Loads all data from the specified files into a `SensorData` objects and yields them separately.

    This function builds the data and meta data paths based on the configuration.
    It automatically detects whether the data is split into multiple chunks, loads and yields each
    chunk of data-metadata pair separately as `SensorData` object.

    Args:
        resample_size (Optional[int], optional):
            Size to which the loaded time series should be resampled.
            If `None`, no resampling is performed. Defaults to `None`.
        normalize (Optional[TNorm], optional):
            Normalization method to apply to the loaded time series.
            If `None`, no normalization is performed. Defaults to `None`.

    Yields:
        SensorData: The next data chunk as a `SensorData` object.
    """


@overload
def load_typed(
    datapath: str,
    *,
    resample_size: int | None = None,
    normalize: TNorm | None = None,
) -> Generator[SensorData, None, None]:
    """
    Loads all data from the specified files into a `SensorData` objects and yields them separately.

    This function takes in data file path and generates the meta data file path based on this data file path.
    It automatically detects whether the data is split into multiple chunks, loads and yields each
    chunk of data-metadata pair separately as `SensorData` object.

    Note, there exists special placeholder for the chunk number in the file paths:

    - `%(chunk)` for any series of digits representable by the Regex pattern `[0-9]+` and
    - `%(chunk?)` as an optional chunking placeholder representable by the Regex pattern `(_?[0-9]+)?`.

    Args:
        datapath (str): The path(s) where the data file(s) are located.
        resample_size (Optional[int], optional):
            Size to which the loaded time series should be resampled.
            If `None`, no resampling is performed. Defaults to `None`.
        normalize (Optional[TNorm], optional):
            Normalization method to apply to the loaded time series.
            If `None`, no normalization is performed. Defaults to `None`.

    Yields:
        SensorData: The next data chunk as a `SensorData` object.
    """


@overload
def load_typed(
    datapath: str,
    metapath: str,
    *,
    resample_size: int | None = None,
    normalize: TNorm | None = None,
) -> Generator[SensorData, None, None]:
    """
    Loads all data from the specified files into a `SensorData` objects and yields them separately.

    This function takes in the paths to data and metadata files.
    It automatically detects whether the data is split into multiple chunks, loads and yields each
    chunk of data-metadata pair separately as `SensorData` object.

    Note, there exists special placeholder for the chunk number in the file paths:

    - `%(chunk)` for any series of digits representable by the Regex pattern `[0-9]+` and
    - `%(chunk?)` as an optional chunking placeholder representable by the Regex pattern `(_?[0-9]+)?`.

    Args:
        datapath (str): The path(s) where the data file(s) are located.
        metapath (str): The path(s) where the metadata file(s) are located.
        resample_size (Optional[int], optional):
            Size to which the loaded time series should be resampled.
            If `None`, no resampling is performed. Defaults to `None`.
        normalize (Optional[TNorm], optional):
            Normalization method to apply to the loaded time series.
            If `None`, no normalization is performed. Defaults to `None`.

    Yields:
        SensorData: The next data chunk as a `SensorData` object.
    """


def load_typed(
    datapath: str | None = None,
    metapath: str | None = None,
    *,
    resample_size: int | None = None,
    normalize: TNorm | None = None,
) -> Generator[SensorData, None, None]:
    """
    Loads all data from the specified files into a `SensorData` objects and yields them separately.

    This function takes in one or both of the paths to data and metadata files.
    It automatically detects whether the data is split into multiple chunks, loads and yields each
    chunk of data-metadata pair separately as `SensorData` object.

    Note, there exists special placeholder for the chunk number in the file paths:

    - `%(chunk)` for any series of digits representable by the Regex pattern `[0-9]+` and
    - `%(chunk?)` as an optional chunking placeholder representable by the Regex pattern `(_?[0-9]+)?`.

    Args:
        datapath (Optional[str], optional):
            The path(s) where the data file(s) are located.
            If `None`, the path will be build from the configuration. Defaults to `None`.
        metapath (Optional[str], optional):
            The path(s) where the metadata file(s) are located.
            If `None`, the path will be build based on the `datapath`. Defaults to `None`.
        resample_size (Optional[int], optional):
            Size to which the loaded time series should be resampled.
            If `None`, no resampling is performed. Defaults to `None`.
        normalize (Optional[TNorm], optional):
            Normalization method to apply to the loaded time series.
            If `None`, no normalization is performed. Defaults to `None`.

    Yields:
        SensorData: The next data chunk as a `SensorData` object.
    """
    logger = getLogger(_IMPORT_LOGGER_NAME)

    for data, metadata in load(datapath, metapath):  # type: ignore[arg-type]
        if Config.LOAD_METADATA_ONLY:
            yield SensorData(np.array([[]] * len(metadata)), metadata)
            continue

        out_type: str = Config.OUT_TYPE
        if datapath is not None:
            _, out_type = os.path.splitext(datapath)
            out_type = out_type[1:]  # remove leading dot

        # convert data to numpy array
        np_data = _convert_data_to_numpy(data, out_type.lower())

        # create container with a SensorData object
        data = SensorDataContainer(SensorData(np_data, metadata))

        # resample data if desired
        if resample_size is not None:
            logger.info(f"Resampling data to size {resample_size}.")
            Resample(data, logger=logger).run(resample_size)

        # normalize data if desired
        if normalize is not None:
            logger.info(f"Normalizing data with '{normalize}' normalization.")
            Normalize(data, logger=logger).run(normalize)

        # return the SensorData object
        yield data.get_sensor_data()


def _convert_data_to_numpy(data: Any, data_type: str) -> np.ndarray:
    """
    Converts the `data` to a NumPy array, according to the specified `data_type`.

    Args:
        data (Any): The data to be converted.
        data_type (str): The data type of the provided `data`.

    Raises:
        ValueError: If an unsupported `data_type` is specified.
        NotImplementedError: If `data_type` is not implemented yet.

    Returns:
        np.ndarray: A NumPy array representation of the `data`.
    """
    if data_type == "hdf5":
        raise NotImplementedError
    if data_type == "asdf":
        raise NotImplementedError
    if data_type == "pd":
        raise NotImplementedError
    if data_type == "nc":
        raise NotImplementedError
    if data_type == "npy":
        # nothing to convert
        return np.array(data)
    if data_type == "pt":
        return data.numpy()
    if data_type == "pkl":
        # should already be a numpy array
        return np.array(data)
    if data_type == "list":
        raise NotImplementedError

    raise ValueError(f"Invalid data type: '{data_type}'")


@overload
def load() -> Generator[tuple[Any, list[dict[str, Any]]], None, None]:
    """
    Loads all data from the specified files as they were exported and yields them separately.

    This function builds the data and meta data paths based on the configuration.
    It automatically detects whether the data is split into multiple chunks, loads and yields each
    chunk of data-metadata pair separately. No further data conversion or transformation is performed.

    Yields:
        Tuple[Any, List[Dict[str, Any]]]: The next data and metadata chunk.
    """


@overload
def load(datapath: str) -> Generator[tuple[Any, list[dict[str, Any]]], None, None]:
    """
    Loads all data from the specified files as they were exported and yields them separately.

    This function takes in data file path and generates the meta data file path based on this data file path.
    It automatically detects whether the data is split into multiple chunks, loads and yields each
    chunk of data-metadata pair separately. No further data conversion or transformation is performed.

    Note, there exists special placeholder for the chunk number in the file paths:

    - `%(chunk)` for any series of digits representable by the Regex pattern `[0-9]+` and
    - `%(chunk?)` as an optional chunking placeholder representable by the Regex pattern `(_?[0-9]+)?`.

    Args:
        datapath (str): The path(s) where the data file(s) are located.

    Yields:
        Tuple[Any, List[Dict[str, Any]]]: The next data and metadata chunk.
    """


@overload
def load(datapath: str, metapath: str) -> Generator[tuple[Any, list[dict[str, Any]]], None, None]:
    """
    Loads all data from the specified files as they were exported and yields them separately.

    This function takes in the paths to data and metadata files.
    It automatically detects whether the data is split into multiple chunks, loads and yields each
    chunk of data-metadata pair separately. No further data conversion or transformation is performed.

    Note, there exists special placeholder for the chunk number in the file paths:

    - `%(chunk)` for any series of digits representable by the Regex pattern `[0-9]+` and
    - `%(chunk?)` as an optional chunking placeholder representable by the Regex pattern `(_?[0-9]+)?`.

    Args:
        datapath (str): The path(s) where the data file(s) are located.
        metapath (str): The path(s) where the metadata file(s) are located.

    Yields:
        Tuple[Any, List[Dict[str, Any]]]: The next data and metadata chunk.
    """


def load(
    datapath: str | None = None,
    metapath: str | None = None,
) -> Generator[tuple[Any, list[dict[str, Any]]], None, None]:
    """
    Loads all data from the specified files as they were exported and yields them separately.

    This function takes in one or both of the paths to data and metadata files.
    It automatically detects whether the data is split into multiple chunks, loads and yields each
    chunk of data-metadata pair separately. No further data conversion or transformation is performed.

    Note, there exists special placeholder for the chunk number in the file paths:

    - `%(chunk)` for any series of digits representable by the Regex pattern `[0-9]+` and
    - `%(chunk?)` as an optional chunking placeholder representable by the Regex pattern `(_?[0-9]+)?`.

    Args:
        datapath (Optional[str], optional):
            The path(s) where the data file(s) are located.
            If `None`, the path will be build from the configuration. Defaults to `None`.
        metapath (Optional[str], optional):
            The path(s) where the metadata file(s) are located.
            If `None`, the path will be build based on the `datapath`. Defaults to `None`.

    Yields:
        Tuple[Any, List[Dict[str, Any]]]: The next data and metadata chunk.
    """
    data_meta_filepath_pairs, import_type = _get_file_pairs(datapath, metapath)

    for data_file, meta_file in data_meta_filepath_pairs:
        yield _load_file_pair(data_file, meta_file, import_type)


def _get_file_pairs(datapath: str | None, metapath: str | None) -> tuple[list[tuple[str | None, str]], str]:
    """
    Returns file path pairs of data and metadata files and the data type from data file paths.

    This functions return all data and metadata file paths of all chunks based on the given paths
    as well as the data type of the data files.

    Args:
        datapath (Optional[str], optional):
            The path(s) where the data file(s) are located.
            If `None`, the path will be build from the configuration. Defaults to `None`.
        metapath (Optional[str], optional):
            The path(s) where the metadata file(s) are located.
            If `None`, the path will be build based on the `datapath`. Defaults to `None`.

    Raises:
        ValueError: If either the data files, metadata files, or both are not readable.

    Returns:
        Tuple[List[Tuple[Optional[str], str]], str]:
            A tuple containing a list of file path pairs and the data type of the data file(s).
            The file path pairs are tuples of the data file path and the metadata file path.
    """
    logger = getLogger(_IMPORT_LOGGER_NAME)

    if datapath is None:
        data_dir = Config.OUT_DIR
        data_filename_without_ext = f"{Config.OUT_NAME}(_[0-9]+)?"
        data_file_extension = Config.OUT_TYPE
        data_filename_with_ext = f"{data_filename_without_ext}\\.{data_file_extension}"
    else:
        data_dir, data_filename_with_ext = os.path.split(datapath)
        data_filename_with_ext = (
            re.escape(data_filename_with_ext)
            .replace("%\\(chunk\\)", "[0-9]+")
            .replace("%\\(chunk\\?\\)", "(_?[0-9]+)?")
        )
        data_filename_without_ext, data_file_extension = os.path.splitext(data_filename_with_ext)
        data_file_extension = data_file_extension[1:]  # remove leading dot

    if metapath is None:
        meta_dir = data_dir
        meta_filename_with_ext = f"{data_filename_without_ext}-meta\\.pkl"
    else:
        meta_dir, meta_filename_with_ext = os.path.split(metapath)
        meta_filename_with_ext = (
            re.escape(meta_filename_with_ext)
            .replace("%\\(chunk\\)", "[0-9]+")
            .replace("%\\(chunk\\?\\)", "(_?[0-9]+)?")
        )

    logger.debug(f"Data directory: {data_dir}")
    logger.debug(f"Meta data directory: {meta_dir}")
    logger.debug(f"Data filename: {data_filename_with_ext}")
    logger.debug(f"Meta data filename: {meta_filename_with_ext}")

    data_files: list[str] = []
    meta_files = sorted(os.path.join(meta_dir, f) for f in os.listdir(meta_dir) if re.match(meta_filename_with_ext, f))
    if not Config.LOAD_METADATA_ONLY:
        data_files = sorted(
            [os.path.join(data_dir, f) for f in os.listdir(data_dir) if re.match(data_filename_with_ext, f)]
        )
        logger.info(f"{len(data_files)} data files and {len(meta_files)} meta data files found.")
        logger.debug(f"Data files: {data_files}")
        logger.debug(f"Meta data files: {meta_files}")
        assert len(data_files) == len(meta_files), "Number of data and meta files do not match."
    else:
        logger.info(f"{len(meta_files)} meta data files found.")
        logger.debug(f"Meta data files: {meta_files}")

    assert len(meta_files) > 0, "No meta data files found."

    if not are_files_readable(data_files, logger) or not are_files_readable(meta_files, logger):
        raise ValueError("Data/Metadata files are not readable.")

    return list(zip_longest(data_files, meta_files)), data_file_extension.lower()


def _load_file_pair(datapath: str | None, metapath: str, import_type: str) -> tuple[Any, list[dict[str, Any]]]:
    """
    Loads data and metadata from separate files into Python objects.

    This function takes in paths to a data file (at `datapath`) and a metadata file
    (at `metapath`), along with the data import type (`import_type`). It uses this
    information to load each file separately, using one of several supported formats.
    The loaded data is returned as a single object (`data`), while the metadata is
    returned as a list of dictionaries.

    Args:
        datapath (Optional[str]):
            The path where the data file is located. If `None`, no data is loaded.
        metapath (str): The path where the metadata file is located.
        import_type (str): The import type to perform on the data file.

    Raises:
        ValueError: If an unsupported `data_type` is specified.
        NotImplementedError: If `data_type` is not implemented yet.

    Returns:
        Tuple[Any, List[Dict[str, Any]]]: A tuple containing the loaded data and metadata.
    """
    logger = getLogger(_IMPORT_LOGGER_NAME)

    data: Any = None
    if datapath:
        # load data
        logger.info(f"Loading data from '{datapath}'...")
        if import_type == "hdf5" or import_type == "asdf" or import_type == "pd" or import_type == "nc":
            raise NotImplementedError
        if import_type == "npy":
            data = np.load(datapath)
        elif import_type == "pt":
            import torch  # load on demand

            data = torch.load(datapath)
        elif import_type == "pkl":
            with open(datapath, "rb") as f:
                data = pickle.load(f)
        elif import_type == "list":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid data type: '{import_type}'")

    # load meta data
    logger.info(f"Loading meta data from '{metapath}'...")
    metadata: list[dict[str, Any]]
    with open(metapath, "rb") as f:
        metadata = pickle.load(f)

    return data, metadata
