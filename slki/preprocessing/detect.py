# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
from collections.abc import Iterable
from datetime import timedelta
from itertools import compress
from typing import Any

import numpy as np
from tslearn.preprocessing import TimeSeriesResampler
from typing_extensions import override

from ..config import Config
from ..data.sensor_data import SensorData
from ..data.sensor_data_item import SensorDataItem
from ..utils import dict_to_object_str
from .stage import Stage


class DetectSignal(Stage):
    """
    A preprocessing stage to detect and trim a signal.

    A preprocessing stage that trims leading and trailing silence from one or more time series,
    checks for signal detection failures, and optionally removes these failures/invalid signals.

    This class provides methods to detect signals in time series data, trim leading and trailing silence,
    and update metadata accordingly.
    """

    def run(
        self,
        window_size: int = 100,
        var_threshold: float = 0.05,
        remove_unrecognized_samples: bool = True,
        neutral_element: float = 0,
    ) -> None:
        """
        Runs the signal detection stage.

        This method applies leading and trailing silence trimming to one or more time series,
        checks for signal detection failures, and optionally removes these failures/invalid signals.
        The resulting data is then updated with the corrected sample lengths.

        Args:
            window_size (int): The size of the sliding window used for variance calculation. Defaults to 100.
            var_threshold (float): The minimum variance threshold above which a time series is considered
                signal-rich. Defaults to 0.05.
            remove_unrecognized_samples (bool): Whether to skip samples with invalid signals detected during
                processing. Defaults to True.
            neutral_element (int | float): The value used to replace leading and trailing silence segments in
                the trimmed time series. Defaults to 0.
        """
        self._window_size = window_size
        self._var_threshold = var_threshold
        self._neutral_element = neutral_element

        self.apply(self._trim_signal, "Detecting Signal")
        self._post_action_check_signal_detection_result(remove_unrecognized_samples)
        if remove_unrecognized_samples:
            self._post_action_remove_unrecognized_samples()
        self._post_action_fix_sample_lengths()

    @override
    def run_default(self) -> None:
        self.run(Config.WINDOW_SIZE, Config.VAR_THRESHOLD, Config.REMOVE_UNRECOGNIZED_SAMPLES)

    def _trim_signal(self, data: np.ndarray, metadata: list[dict[str, Any]], **kwargs) -> np.ndarray:
        """
        Detecting the signal in the time series.

        This method trims leading and trailing silence from time series by checking the variance of
        the sliding window view of the signal data.

        Args:
            data (np.ndarray): A 1D or 2D NumPy array.
                1D with just a single time series or 2D where each row represents a single time series.
            metadata (List[Dict[str, Any]]): A list of dictionaries containing metadata for each time series.
            **kwargs: Additional arguments will be ignored.

        Returns:
            np.ndarray: The input `data` with leading and trailing silence trimmed from each time series.
        """
        # NOTE: data can be a single time series or an array of time series
        signal_window = np.lib.stride_tricks.sliding_window_view(data, self._window_size, axis=-1)
        var = np.var(signal_window, axis=-1)
        above_threshold = var > self._var_threshold

        for idx, (thresholds, meta) in enumerate(zip(above_threshold, metadata, strict=False)):
            indices, *_ = np.nonzero(thresholds)
            if len(indices) < 1:
                meta["signal_detected"] = False
                continue

            start: int = indices[0]
            end: int = indices[-1]
            length = end - start

            # adjust start and end time to detected signal
            # if self.data_container.resampled:
            #     # since the signal is already resample, try to adjust the start and end time as good as possible
            #     sample_length = meta["sample_length"]
            #     delta = meta["start_end"] - meta["start_time"]
            #     meta["start_time"] += delta * (start / sample_length)
            #     meta["start_end"] -= delta * ((sample_length - end) / sample_length)
            # else:
            # use samples and sample rate to calculate the "new" start and end times correctly
            sample_rate_in_hz: float = meta["sample_rate_in_hz"]
            meta["start_time"] += timedelta(seconds=(start + 1) / sample_rate_in_hz)
            meta["end_time"] -= timedelta(seconds=(meta["sample_length"] - (end + 1)) / sample_rate_in_hz)

            # trim signal
            data[idx, :length] = data[idx, start:end]
            data[idx, length:] = self._neutral_element
            meta["sample_length"] = length

        return data

    def _post_action_check_signal_detection_result(self, remove_unrecognized_samples: bool) -> None:
        """
        Logs warnings and debug information about signal detection failures.

        This method checks if there are any failed signal detections in the data container.
        If so, it logs a warning message indicating the number of failed detections. It also logs a
        debug message listing the origins of the failed signals for further investigation.

        Args:
            remove_unrecognized_samples (bool): Whether to indicate that failed samples should be skipped, or kept.
        """
        # log warning if any signals detections failed.
        total = len(self.data_container)
        detected = sum(self.data_container.get_meta("signal_detected", True))
        if total != detected:
            msg = f"{total - detected} of {total} signal detections failed. "
            if remove_unrecognized_samples:
                msg += "Skip these samples."
            else:
                msg += "Skip signal detection and leading/tailing silence trimming for these samples but them."
            self.logger.warning(msg)

            msg = "Signal detection failed for the signals:"
            for meta in self.data_container.get_metadata():
                if not meta.get("signal_detected", True):
                    origin: dict[str, Any] = meta.get("origin", None)  # type: ignore[assignment]
                    origin_str = dict_to_object_str(origin, "Origin") or '"unknown origin"'
                    msg += f"\norigin={origin_str}"
            self.logger.debug(msg)

    def _post_action_remove_unrecognized_samples(self) -> None:
        """Removes samples where signal detection failed from the data container."""
        flags: list[bool] = self.data_container.get_meta("signal_detected", True)
        if isinstance(self.data_container.data, SensorData):
            sensor_data = self.data_container.data
            sensor_data.data = sensor_data.data[flags]
            sensor_data.meta = list(compress(sensor_data.meta, flags))
        else:
            self.data_container.data = list(compress(self.data_container.data, flags))
        # check if the number of data samples is consistent with the meta data
        assert len(self.data_container) == len(self.data_container.get_metadata())

    def _post_action_fix_sample_lengths(self) -> None:
        """
        Fixes sample lengths and resamples signals (if necessary).

        Fixing sample lengths by cutting or expanding the singal array. If the data is already
        resampled, the signals will be resample to the new desired sample length accordingly.

        Depending on the type of data in `self.data_container`, this method calls either
        `_post_action_fix_sample_lengths_list` or `_post_action_fix_sample_lengths_matrix`
        to perform the actual fixup. The choice is based on whether a single list of SensorDataItems
        or a matrix-like SensorData object is present.
        """
        if isinstance(self.data_container.data, SensorData):
            self.data_container.data = self._post_action_fix_sample_lengths_matrix(self.data_container.data)
        else:
            self.data_container.data = self._post_action_fix_sample_lengths_list(self.data_container.data)

    def _post_action_fix_sample_lengths_matrix(self, sensor_data: SensorData) -> SensorData:
        """
        Fixes sample lengths and resamples signals (if necessary) for a SensorData object.

        Args:
            sensor_data (SensorData): The SensorData object with the signals to fix.

        Returns:
            SensorData: The updated SensorData object with valid signal lengths.
        """
        signal_lengths = self.data_container.sample_lengths
        # drop all signals which are not valid (are to short or to long)
        valid_flags = self._get_valid_signal_length_flags(signal_lengths, sensor_data.meta)
        sensor_data.data = sensor_data.data[valid_flags]
        sensor_data.meta = list(compress(sensor_data.meta, valid_flags))
        signal_lengths = list(compress(signal_lengths, valid_flags))
        # fix signal sample lengths
        max_length = max(signal_lengths)
        sensor_data.data = sensor_data.data[:, :max_length]
        # fix sample (resample) if necessary
        if self.data_container.resampled:
            for data_item, signal_length in zip(sensor_data, signal_lengths, strict=False):
                resampler = TimeSeriesResampler(max_length)
                data_item.data[:] = resampler.fit_transform(data_item.data[:signal_length]).flatten()
        # return fixed data
        return sensor_data

    def _post_action_fix_sample_lengths_list(self, sensor_data_items: list[SensorDataItem]) -> list[SensorDataItem]:
        """
        Fixes sample lengths and resamples signals (if necessary) for a list of SensorDataItems.

        Args:
            sensor_data_items (List[SensorDataItem]): The list of data items with the signals to fix.

        Returns:
            List[SensorDataItem]: The updated list of SensorDataItems with valid signal lengths.
        """
        signal_lengths = self.data_container.sample_lengths
        # drop all signals which are not valid (are to short or to long)
        valid_flags = self._get_valid_signal_length_flags(signal_lengths, (item.meta for item in sensor_data_items))
        sensor_data_items = list(compress(sensor_data_items, valid_flags))
        signal_lengths = list(compress(signal_lengths, valid_flags))
        # fix signal sample lengths and resample signal if necessary
        for data_item, signal_length in zip(sensor_data_items, signal_lengths, strict=False):
            if self.data_container.resampled:
                resampler = TimeSeriesResampler(len(data_item.data))
                data_item.data[:] = resampler.fit_transform(data_item.data[:signal_length]).flatten()
            else:
                # just cut the signal
                data_item.data = data_item.data[:signal_length]
        # return fix data
        return sensor_data_items

    def _get_valid_signal_length_flags(
        self,
        signal_lengths: list[int],
        metadata: Iterable[dict[str, Any]],
        disable_warning: bool = False,
    ) -> np.ndarray:
        """
        Returns a mask (boolean array) indicating whether each signal length is valid.

        A signal length is considered valid if it is greater than or equal to the lower bound
        and less than or equal to the upper bound.

        Args:
            signal_lengths (List[int]): The list of signal lengths to check.
            metadata (Iterable[Dict[str, Any]]): The metadata associated with each signal length.
            disable_warning (bool, optional): Whether to disable the warning message. Defaults to False.

        Returns:
            np.ndarray: Mask (boolean array) which indicats whether each signal length is valid.
        """
        lengths = np.array(signal_lengths)
        flags = np.array([True] * len(signal_lengths))
        flags[(Config.SAMPLE_LENGTH_LOWER_BOUND or float("-inf")) > lengths] = False
        # detecting signals can result only in smaller signals therefore is the upper bound check irrelevant
        # flags[lengths > (Config.SAMPLE_LENGTH_UPPER_BOUND or float("inf"))]
        if not disable_warning and not flags.all():
            self.logger.warning(
                f'{np.count_nonzero(~flags)} of {len(signal_lengths)} "detected" signals are shorter'
                f" than {Config.SAMPLE_LENGTH_LOWER_BOUND}. Drop these signals.",
            )
            msg = "Dropped signals:"
            for meta in compress(metadata, ~flags):
                origin: dict[str, Any] = meta.get("origin", None)  # type: ignore[assignment]
                origin_str = dict_to_object_str(origin, "Origin") or '"unknown origin"'
                msg += f"\norigin={origin_str}"
            self.logger.debug(msg)
        return flags
