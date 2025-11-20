# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
from collections.abc import Generator
from itertools import islice
from logging import getLogger
import math

from tqdm import tqdm

from ..config import Config
from ..data import SensorData, SensorDataContainer, SensorDataItem
from ..data.hdf5_data_reader import HDF5DataReader
from ..preprocessing import get_stage_cls
from ..utils.path import find_import_files
from ..utils.tqdm import get_tqdm_kwargs


class Pipeline:
    """
    A pipeline class that encapsulates data import and processing functionality.

    This class provides methods to load sensor data from various sources, process it through a series of stages,
    and return the final processed data. The pipeline is designed to be customizable through configuration options.
    """

    def __init__(self) -> None:
        """Initializes a new pipeline class object."""
        self.data: SensorDataContainer
        """The sensor data container."""
        self._logger = getLogger("slki.pipeline")
        """A logger instance used for logging events during the pipeline execution."""
        self._num_chunks: int | None = None
        """An estimation of the number of data chunks."""

    def get_data_container(self) -> SensorDataContainer:
        """
        Retrieves current state of the imported and processed sensor data as sensor data container.

        Returns:
            SensorDataContainer: The current state of the imported and processed sensor data.
        """
        return self.data

    def get_sensor_data(self) -> SensorData:
        """
        Retrieves current state of the imported and processed sensor data as sensor data.

        Returns:
            SensorData: The current state of the imported and processed sensor data.
        """
        return self.data.get_sensor_data()

    def get_sensor_data_items(self) -> list[SensorDataItem]:
        """
        Retrieves current state of the imported and processed sensor data as list of sensor data items.

        Returns:
            SensorData: The current state of the imported and processed sensor data.
        """
        return self.data.get_sensor_data_items()

    def get_number_of_chunks_estimation(self) -> int | None:
        """
        Get the estimated number of data chunks.

        Returns:
            Optional[int]: Estimated number of data chunks. None if data loading chunking is not enabled.
        """
        return self._num_chunks

    def import_data(self) -> Generator[SensorDataContainer, None, None]:
        """
        Imports sensor data from various sources as chunks.

        This method loads data from files or other sources based on the configuration options,
        analyzes and fetches the data generator, and then chunks the data if desired.
        It uses a progress bar to display the loading process.

        The method yields each chunk of imported data as it is loaded, allowing the caller
        to receive and further process the data without concerning about memory (RAM).

        Raises:
            ValueError: If the import type is not recognized or supported.

        Yields:
            SensorDataContainer: The next chunk of imported data.
        """
        # data source files
        filepaths = list(find_import_files(Config.IMPORT_SOURCES))

        # analyse data and fetch data generator
        if Config.IMPORT_TYPE == "hdf5":
            data_generator, max_signal_size = HDF5DataReader().load(filepaths)
        else:
            raise ValueError(f"Invalid import type: '{Config.IMPORT_TYPE}'")

        # prepare chunking if desired
        stop = None if Config.CHUNK_POINT != "onload" or Config.CHUNK_SIZE is None else Config.CHUNK_SIZE
        chunking = stop is not None and max_signal_size > stop
        self._num_chunks = 1 if not chunking else math.ceil(max_signal_size / stop)  # type: ignore[operator]
        tqdm_desc = "Loading HDF5 data"
        if chunking:
            tqdm_desc += f" [{{0}}/{self._num_chunks}]"

        n = 1
        while True:
            # prepare progress bar
            total: int
            if not chunking or stop is None:
                total = max_signal_size
            elif n < self._num_chunks:
                total = stop
            else:
                total = max_signal_size - stop * (self._num_chunks - 1)  # last chunk size estimation (might be smaller)
            tqdm_generator = tqdm(data_generator, total=total, **get_tqdm_kwargs(desc=tqdm_desc.format(n), leave=False))

            # fetch (next chunked) data
            data: list[SensorDataItem] = list(islice(tqdm_generator, stop))
            n += 1

            # cleanup progress bar
            if Config.TQDM_OPTIONS.get("leave", True) and len(data) > 0:
                tqdm_generator.disable = False
                tqdm_generator.leave = True
                tqdm_generator.total = len(data)
                tqdm_generator.n = tqdm_generator.total
                tqdm_generator.refresh()
            tqdm_generator.close()

            # check if there is any data, otherwise data import is done
            if not data:
                break  # break while and exit function

            # convert data to desired form and precision type and return it
            self.data = SensorDataContainer(data)
            self.data = self.data.astype(Config.PRECISION)
            self._logger.debug(f"Imported {len(data)} items")
            yield self.data

    def run(self) -> SensorDataContainer:
        """
        Runs the pipeline and returns the final state of the processed sensor data.

        Returns:
            SensorDataContainer: The fully processed sensor data.
        """
        for _ in self.run_generator():
            continue
        return self.data

    def run_generator(self) -> Generator[SensorDataContainer, None, None]:
        """
        Runs a generator to process the data while yielding the result of each processing stage.

        This method starts by checking if metadata-only loading is enabled. If so, it skips data preprocessing.
        Otherwise, it yields the raw imported data and then iterates over each processing stage defined in the configuration.
        For each stage, it runs the corresponding preprocessing function on the data and yields the result of that stage.

        Yields:
            SensorDataContainer: The next processing stage result.
        """  # noqa: E501
        self._logger.debug("Start preprocessing the data...")

        # check if any data should be processed
        if Config.LOAD_METADATA_ONLY:
            self._logger.warning("LOAD_METADATA_ONLY is set to True. Therefore, data preprocessing will be skiped.")
            return

        # raw data
        yield self.data

        # check if there is any data available
        if len(self.data) < 1:
            self._logger.error("No data to process.")
            return

        # check if there are any stages defined
        if not Config.STAGES:
            self._logger.warning("No preprocessing stages defined. Data will be returned as is.")
            self._logger.info("Preprocessing skiped")
            return

        # run preprocessing stages
        with tqdm(Config.STAGES, **get_tqdm_kwargs(desc="Processing")) as pbar:
            for stage_name in pbar:
                pbar.set_postfix({"stage": stage_name}, refresh=True)
                self._logger.debug(f"Running stage '{stage_name}'")

                stage_cls = get_stage_cls(stage_name)
                stage = stage_cls(self.data)
                stage.run_default()

                self._logger.debug(f"Stage '{stage_name}' done")
                self.data = stage.data_container  # should already be the same since it is a reference
                yield self.data
            pbar.set_postfix({}, refresh=True)

        self._logger.debug("Preprocessing: Done.")
