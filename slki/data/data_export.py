# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
from collections.abc import Sequence
from logging import Logger, getLogger
import math
import os
import pickle
from typing import Any, overload

import numpy as np

from ..config import Config, TChunking
from .sensor_data import SensorData
from .sensor_data_container import SensorDataContainer
from .sensor_data_item import SensorDataItem


@overload
def dump(data: SensorData) -> None:
    """
    Exports the given data.

    This method prepares the input data for export, detects the desired chunk size based
    on the specified chunking configuration, and writes the data to a file (or files)
    in a format suitable for future analysis and processing.

    Args:
        data (SensorData): The data to be exported.
    """


@overload
def dump(
    data: SensorData,
    *,
    chunk_number: int,
    chunk_digits: int | None = None,
) -> None:
    """
    Exports the given data.

    This method prepares the input data for export, detects the desired chunk size based
    on the specified chunking configuration, and writes the data to a file (or files)
    in a format suitable for future analysis and processing.

    Args:
        data (SensorData): The data to be exported.
        chunk_number (int):
            The number of the current data chunk which should be exported.
            This number will be used to create a reasonable file name.
        chunk_digits (Optional[int], optional):
           The number of digits to use for formatting the chunk number.
           If `None`, no formatting will be applied.
    """


@overload
def dump(
    data: SensorData,
    *,
    chunk_point: TChunking,
    chunk_size: int,
) -> None:
    """
    Exports the given data.

    This method prepares the input data for export, detects the desired chunk size based
    on the specified chunking configuration, and writes the data to a file (or files)
    in a format suitable for future analysis and processing.

    Args:
        data (SensorData): The data to be exported.
        chunk_point (TChunking): The chunking point. The moment when the data should be chunked.
        chunk_size (int): The chunk size. The number of data items in each chunk.
    """


@overload
def dump(data: list[SensorDataItem]) -> None:
    """
    Exports the given data.

    This method prepares the input data for export, detects the desired chunk size based
    on the specified chunking configuration, and writes the data to a file (or files)
    in a format suitable for future analysis and processing.

    Args:
        data (List[SensorDataItem]): The data to be exported.
    """


@overload
def dump(
    data: list[SensorDataItem],
    *,
    chunk_number: int,
    chunk_digits: int | None = None,
) -> None:
    """
    Exports the given data.

    This method prepares the input data for export, detects the desired chunk size based
    on the specified chunking configuration, and writes the data to a file (or files)
    in a format suitable for future analysis and processing.

    Args:
        data (List[SensorDataItem]): The data to be exported.
        chunk_number (int):
            The number of the current data chunk which should be exported.
            This number will be used to create a reasonable file name.
        chunk_digits (Optional[int], optional):
           The number of digits to use for formatting the chunk number.
           If `None`, no formatting will be applied.
    """


@overload
def dump(
    data: list[SensorDataItem],
    *,
    chunk_point: TChunking,
    chunk_size: int,
) -> None:
    """
    Exports the given data.

    This method prepares the input data for export, detects the desired chunk size based
    on the specified chunking configuration, and writes the data to a file (or files)
    in a format suitable for future analysis and processing.

    Args:
        data (List[SensorDataItem]): The data to be exported.
        chunk_point (TChunking): The chunking point. The moment when the data should be chunked.
        chunk_size (int): The chunk size. The number of data items in each chunk.
    """


@overload
def dump(data: SensorDataContainer) -> None:
    """
    Exports the given data.

    This method prepares the input data for export, detects the desired chunk size based
    on the specified chunking configuration, and writes the data to a file (or files)
    in a format suitable for future analysis and processing.

    Args:
        data (SensorDataContainer): The data to be exported.
    """


@overload
def dump(
    data: SensorDataContainer,
    *,
    chunk_number: int,
    chunk_digits: int | None = None,
) -> None:
    """
    Exports the given data.

    This method prepares the input data for export, detects the desired chunk size based
    on the specified chunking configuration, and writes the data to a file (or files)
    in a format suitable for future analysis and processing.

    Args:
        data (SensorDataContainer): The data to be exported.
        chunk_number (int):
            The number of the current data chunk which should be exported.
            This number will be used to create a reasonable file name.
        chunk_digits (Optional[int], optional):
           The number of digits to use for formatting the chunk number.
           If `None`, no formatting will be applied.
    """


@overload
def dump(
    data: SensorDataContainer,
    *,
    chunk_point: TChunking,
    chunk_size: int,
) -> None:
    """
    Exports the given data.

    This method prepares the input data for export, detects the desired chunk size based
    on the specified chunking configuration, and writes the data to a file (or files)
    in a format suitable for future analysis and processing.

    Args:
        data (SensorDataContainer): The data to be exported.
        chunk_point (TChunking): The chunking point. The moment when the data should be chunked.
        chunk_size (int): The chunk size. The number of data items in each chunk.
    """


@overload
def dump(data: np.ndarray, meta: Sequence[dict[str, Any]]) -> None:
    """
    Exports the given data.

    This method prepares the input data for export, detects the desired chunk size based
    on the specified chunking configuration, and writes the data to a file (or files)
    in a format suitable for future analysis and processing.

    Args:
        data (np.ndarray): The data to be exported.
        meta (Sequence[Dict[str, Any]]): The metadata associated with the data to be exported.
    """


@overload
def dump(
    data: np.ndarray,
    meta: Sequence[dict[str, Any]],
    *,
    chunk_number: int,
    chunk_digits: int | None = None,
) -> None:
    """
    Exports the given data.

    This method prepares the input data for export, detects the desired chunk size based
    on the specified chunking configuration, and writes the data to a file (or files)
    in a format suitable for future analysis and processing.

    Args:
        data (np.ndarray): The data to be exported.
        meta (Sequence[Dict[str, Any]]): The metadata associated with the data to be exported.
        chunk_number (int):
            The number of the current data chunk which should be exported.
            This number will be used to create a reasonable file name.
        chunk_digits (Optional[int], optional):
           The number of digits to use for formatting the chunk number.
           If `None`, no formatting will be applied.
    """


@overload
def dump(
    data: np.ndarray,
    meta: Sequence[dict[str, Any]],
    *,
    chunk_point: TChunking,
    chunk_size: int,
) -> None:
    """
    Exports the given data.

    This method prepares the input data for export, detects the desired chunk size based
    on the specified chunking configuration, and writes the data to a file (or files)
    in a format suitable for future analysis and processing.

    Args:
        data (np.ndarray): The data to be exported.
        meta (Sequence[Dict[str, Any]]): The metadata associated with the data to be exported.
        chunk_point (TChunking): The chunking point. The moment when the data should be chunked.
        chunk_size (int): The chunk size. The number of data items in each chunk.
    """


def dump(
    data: SensorDataContainer | SensorData | list[SensorDataItem] | np.ndarray,
    meta: Sequence[dict[str, Any]] | None = None,
    *,
    chunk_point: TChunking = Config.CHUNK_POINT,
    chunk_size: int | None = Config.CHUNK_SIZE,
    chunk_number: int | None = None,
    chunk_digits: int | None = None,
) -> None:
    """
    Exports the given data.

    This method prepares the input data for export, detects the desired chunk size based
    on the specified chunking configuration, and writes the data to a file (or files)
    in a format suitable for future analysis and processing.

    Args:
        data (SensorDataContainer | SensorData | List[SensorDataItem] | np.ndarray):
            The data to be exported.
        meta (Optional[Sequence[Dict[str, Any]]], optional):
            Metadata associated with the data.
            If the provided data is an NumPy array, the metadata have to be provided separably.
            Otherwise the metadata will be taken from the provided data object.
            Defaults to `None`.
        chunk_point (TChunking, optional): The desired chunking point. Defaults to `Config.CHUNK_POINT`.
        chunk_size (Optional[int], optional):
            The desired chunk size. If None, the data will not be chunked. Defaults to `Config.CHUNK_SIZE`.
        chunk_number (Optional[int], optional):
            When `chunk_point` is "onload", this specifies the number of the current data chunk which should
            be exported. This number will be used to create a reasonable file name.
            Defaults to `None`.
        chunk_digits (Optional[int], optional):
            When `chunk_point` is "onload", this specifies the number of digits to use for formatting the chunk number.
            If `None`, no formatting will be applied. Defaults to `None`.
    """
    os.makedirs(Config.OUT_DIR, exist_ok=True)
    logger = getLogger("slki.data.export")

    # prepare data as `List[SensorDataItem]`
    data = _prepare_data(data, meta)
    logger.debug(f"Exporting {len(data)} data items.")

    # detect desired chunk size
    if chunk_point != "onsave" and chunk_size is not None and chunk_size < len(data):
        _export_data_in_chunks(data, chunk_size, logger)
        return

    # build file path
    chunking_surfix = "" if chunk_number is None else f"_{chunk_number:0>{chunk_digits or 1}d}"
    file_basepath = f"{os.path.join(Config.OUT_DIR, Config.OUT_NAME)}{chunking_surfix}"

    # export (non chunked) data
    _export_data_single_chunk(file_basepath, data, logger)


def _prepare_data(
    data: SensorDataContainer | SensorData | list[SensorDataItem] | np.ndarray,
    meta: Sequence[dict[str, Any]] | None = None,
) -> list[SensorDataItem]:
    """
    Prepares the data for export by interpreting them as a list of `SensorDataItem` objects.

    Args:
        data (SensorDataContainer | SensorData | List[SensorDataItem] | np.ndarray):
            The data to be exported.
        meta (Optional[Sequence[Dict[str, Any]]], optional):
            Metadata associated with the data.
            If the provided data is an NumPy array, the metadata have to be provided separably.
            Otherwise the metadata will be taken from the provided data object.
            Defaults to `None`.

    Raises:
        ValueError: If `data` has an invalid type.
        AssertionError: If `data` is a NumPy array but no metadata are provided.

    Returns:
        List[SensorDataItem]: A list of `SensorDataItem` objects representing the prepared data.
    """
    if isinstance(data, SensorDataContainer):
        return data.get_sensor_data_items()
    if isinstance(data, SensorData):
        return list(iter(data))
    if isinstance(data, np.ndarray):
        assert meta is not None
        return [SensorDataItem(d, m) for d, m in zip(data, meta, strict=False)]
    raise ValueError(f"Export: Invalid data type: '{type(data)}'")


def _export_data_in_chunks(data: list[SensorDataItem], chunk_size: int, logger: Logger) -> None:
    """
    Exports the given sensor data in chunks of a specified size.

    This method divides the `data` into chunks based on the provided `chunk_size`
    and exports each chunk separately.

    Args:
        data (List[SensorDataItem]): The to be chunked and exported.
        chunk_size (int): The size of each chunk.
        logger (Logger): The logger object to log information during the export process.
    """
    num_chunks = math.ceil(len(data) / chunk_size)
    if logger:
        logger.info(f"Data will be chunked into {num_chunks} chunks of size {chunk_size}.")

    for chunk_idx in range(num_chunks):
        chunk_start_idx = chunk_idx * chunk_size
        chunk_data = data[chunk_start_idx : chunk_start_idx + chunk_size]

        # build file path
        chunking_surfix = f"_{(chunk_idx + 1):0>{len(str(num_chunks))}d}"
        file_basepath = f"{os.path.join(Config.OUT_DIR, Config.OUT_NAME)}{chunking_surfix}"

        # export data (single chunk)
        _export_data_single_chunk(file_basepath, chunk_data, logger)


def _export_data_single_chunk(
    file_basepath: str,
    data: list[SensorDataItem],
    logger: Logger | None = None,
) -> None:
    """
    Exports a single chunk of data.

    Args:
        file_basepath (str): The base path for the output files (without file extension).
        data (List[SensorDataItem]): The data to be exported.
        logger (Optional[Logger], optional):
            The logger object to log information during the export process.
            If `None` logging is disabled. Defaults to `None`.
    """
    data_filepath = f"{file_basepath}.{Config.OUT_TYPE}"
    meta_filepath = f"{file_basepath}-meta.pkl"

    if not Config.LOAD_METADATA_ONLY:
        # prepare data as one big matrix (numpy array)
        data_as_one_matrix = SensorData.from_sensor_data_items(data)
        export_data = data_as_one_matrix.get_data()
        export_meta = data_as_one_matrix.get_metadata()

        # save data to file
        _export_data(data_filepath, export_data, logger)
    else:
        export_meta = [item.meta for item in data]

    # save meta data as pickle
    with open(meta_filepath, "wb") as f:
        pickle.dump(export_meta, f)

    if logger:
        logger.info(f"Meta data saved to '{meta_filepath}'")


def _export_data(data_filepath: str, data: np.ndarray, logger: Logger | None = None) -> None:
    """
    Exports the given NumPy array to a file in a format specified by `Config.OUT_TYPE`.

    Args:
        data_filepath (str): The path where the output file should be saved.
        data (np.ndarray): The 2D NumPy array to be exported where each row represents a time series.
        logger (Optional[Logger], optional):
            The logger object to log information during the export process.
            If `None` logging is disabled. Defaults to `None`.

    Raises:
        ValueError: If `Config.OUT_TYPE` is an unsupported format.
        NotImplementedError: If `Config.OUT_TYPE` is not implemented yet.
    """
    # save data in desired format
    if Config.OUT_TYPE == "hdf5" or Config.OUT_TYPE == "asdf" or Config.OUT_TYPE == "pd" or Config.OUT_TYPE == "nc":
        raise NotImplementedError
    if Config.OUT_TYPE == "npy":
        np.save(data_filepath, data)
    elif Config.OUT_TYPE == "pt":
        import torch  # load on demand

        tensor = torch.from_numpy(data)
        torch.save(tensor, data_filepath)
    elif Config.OUT_TYPE == "pkl":
        with open(data_filepath, "wb") as f:
            pickle.dump(data, f)
    elif Config.OUT_TYPE == "list":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid data type: '{Config.OUT_TYPE}'")

    if logger:
        logger.info(f"Data saved to '{data_filepath}'")
