# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
from typing import Any

import noisereduce as nr
import numpy as np
from typing_extensions import override

from ..config import Config
from ..utils import dict_to_object_str
from .stage import Stage


class Denoise(Stage):
    """
    A stage that applies denoising to the input data.

    This stage uses the `noisereduce.reduce_noise` function to reduce noise in the input signal.
    """

    SUPPORTS_ONLY_ITERATIVE: bool = True

    def run(self, sr: int | None = None, **denoise_kwargs) -> None:
        """
        Runs the denoising process.

        Args:
            sr (Optional[int], optional): The sample rate to use for denoising.
                If not provided, it will be taken from the metadata.
            **denoise_kwargs: Additional keyword arguments to pass to `noisereduce.reduce_noise`.
        """
        self._sr_overwrite = sr
        self._denoise_kwargs = denoise_kwargs
        self.apply(self._denoise, "Denoising")

    @override
    def run_default(self) -> None:
        self.run(**Config.DENOISE_OPTIONS)

    def _denoise(self, data: np.ndarray, metadata: list[dict[str, Any]], **kwargs) -> np.ndarray:
        """
        Applies denoising to a single data sample.

        Args:
            data (np.ndarray): The input signal to denoise.
            metadata (List[Dict[str, Any]]): Metadata associated with the input signal.
            **kwargs: Additional arguments will be ignored.

        Returns:
            np.ndarray: The denoised signal.
        """
        # since we are calling run_action with force_iterative=True,
        # data and metadata will always contain exactly one data sample
        signal, meta = data.squeeze(axis=0), metadata[0]
        sr = self._sr_overwrite or meta.get("sample_rate_in_hz", None)
        if sr is None:
            origin: dict[str, Any] = meta.get("origin", None)  # type: ignore[assignment]
            origin_str = dict_to_object_str(origin, "Origin") or '"unknown origin"'
            self.logger.error(f"No sample rate found for {origin_str}. Skip denoising for this time series.")
            return data  # continue
        ext_length = int(2 * sr)
        signal = np.concatenate([np.flip(signal[:ext_length]), signal, np.flip(signal[-ext_length:])])
        signal = nr.reduce_noise(y=signal, sr=sr, **self._denoise_kwargs)
        signal = signal[ext_length:-ext_length]
        return np.expand_dims(signal, axis=0)
