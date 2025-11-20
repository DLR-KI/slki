# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
from abc import ABCMeta, abstractmethod
from logging import Logger, getLogger
from typing import Any

import numpy as np
from typing_extensions import Self

from ..data.sensor_data_container import SensorDataContainer
from ..data.sensor_data_item import SensorDataItem


class Stage(metaclass=ABCMeta):
    """
    Abstract base class for all preprocessing stages.

    This class serves as a base for concrete stage implementations,
    providing methods for running and configuring the stage.
    Concrete stages should subclass this class and implement the required
    methods to perform their specific processing tasks.
    """

    SUPPORTS_ONLY_ITERATIVE: bool = False
    """Whether this stage supports only parallel processing."""

    def __init__(
        self,
        data_container: SensorDataContainer,
        *,
        logger: Logger | None = None,
        force_iterative: bool | None = None,
    ) -> None:
        """
        Initializes a new instance of a preprocessing stage.

        Args:
            data_container (SensorDataContainer): The container holding the data.
            logger (Optional[Logger], optional): An optional logger to use for logging messages.
                If not provided, a default logger will be used.
            force_iterative (Optional[bool], optional): A flag indicating whether the stage should be forced
                to process the data iterative and not in parallel.
        """
        self.data_container = data_container
        self.logger = logger or getLogger("slki.preprocessing")
        self.force_iterative = force_iterative

    def apply(
        self,
        fn,
        desc: str | None = None,
        tqdm_overwrite_kwargs: dict[str, Any] | None = None,
        *,
        force_iterative: bool = False,
    ) -> None:
        """
        Applies a given function to the data container.

        This method wraps the `apply` method of the underlying data container and allows for
        iterative processing. If the stage is configured to force iterative processing, the
        data will be processed iteratively, regardless of the underlying data container's
        capabilities.

        Args:
            fn (Callable): The function to be applied to the data.
            desc (Optional[str], optional): An optional description string for the application process.
            tqdm_overwrite_kwargs (Optional[Dict[str, Any]], optional): Overrideable tqdm kwargs.
            force_iterative (bool, optional): A flag indicating whether iterative processing should be forced.
                Defaults to False.
        """
        iterative = force_iterative or self.force_iterative or self.__class__.SUPPORTS_ONLY_ITERATIVE
        self.data_container.apply(
            fn,
            desc,
            tqdm_overwrite_kwargs,
            force_iterative=iterative,
        )

    @abstractmethod
    def run(self, *args, **kwargs) -> None:
        """
        Abstract method to be implemented by concrete stages.

        This method should perform the actual processing task for this stage.

        Args:
            *args: Any additional arguments that may be desired by the `run` method.
            **kwargs: Additional keyword arguments that may be desired by the `run` method.

        Raises:
            NotImplementedError: If this abstract method is not overridden in a concrete subclass.

        Note:
            Concrete stages must implement this method to provide their specific processing logic.
        """

    def run_default(self) -> None:
        """
        Runs the stage using default configuration options.

        This method simply calls the `run` method, which allows for running the stage with the default configuration.
        """
        self.run()

    def __call__(self, *args, **kwargs) -> None:
        """
        Calls the stage's `run` method.

        This special method allows for calling the stage as a function, which will execute its processing logic.

        Args:
            *args: Any additional arguments that may be desired by the `run` method.
            **kwargs: Additional keyword arguments that may be desired by the `run` method.
        """
        self.run(*args, **kwargs)

    @classmethod
    def from_data_and_meta(
        cls,
        data: np.ndarray,
        meta: dict[str, Any],
        *,
        logger: Logger | None = None,
        force_iterative: bool | None = None,
    ) -> Self:
        """
        Creates and initializes a new instance of a preprocessing stage for a single signal.

        Args:
            data (np.ndarray): Single signal data.
            meta (Dict[str, Any]): Meta data for the single signal.
            logger (Optional[Logger], optional): An optional logger to use for logging messages.
                If not provided, a default logger will be used.
            force_iterative (Optional[bool], optional): A flag indicating whether the stage should be forced
                to process the data iterative and not in parallel.
            **meta_kwargs: Additional metadata key-value pairs.

        Returns:
            Self: Initialized preprocessing stage.
        """
        return cls(
            SensorDataContainer([SensorDataItem(data, meta)]),
            logger=logger,
            force_iterative=force_iterative,
        )

    @classmethod
    def from_data_and_sample_length(
        cls,
        data: np.ndarray,
        sample_length: int,
        *,
        logger: Logger | None = None,
        force_iterative: bool | None = None,
        **meta_kwargs: Any,
    ) -> Self:
        """
        Creates and initializes a new instance of a preprocessing stage for a single signal.

        Args:
            data (np.ndarray): Single signal data.
            sample_length (int): Original signal length. (Not the resampled signal length.)
            logger (Optional[Logger], optional): An optional logger to use for logging messages.
                If not provided, a default logger will be used.
            force_iterative (Optional[bool], optional): A flag indicating whether the stage should be forced
                to process the data iterative and not in parallel.
            **meta_kwargs: Additional metadata key-value pairs.

        Returns:
            Self: Initialized preprocessing stage.
        """
        return cls(
            SensorDataContainer([SensorDataItem.create(data, sample_length=sample_length, **meta_kwargs)]),
            logger=logger,
            force_iterative=force_iterative,
        )

    @classmethod
    def from_none_resampled_data(
        cls,
        data: np.ndarray,
        *,
        logger: Logger | None = None,
        force_iterative: bool | None = None,
        **meta_kwargs: Any,
    ) -> Self:
        """
        Creates and initializes a new instance of a preprocessing stage for a single (not resampled) signal.

        Args:
            data (np.ndarray): Single (not resampled) signal data.
            logger (Optional[Logger], optional): An optional logger to use for logging messages.
                If not provided, a default logger will be used.
            force_iterative (Optional[bool], optional): A flag indicating whether the stage should be forced
                to process the data iterative and not in parallel.
            **meta_kwargs: Additional metadata key-value pairs.

        Returns:
            Self: Initialized preprocessing stage.
        """
        print(cls)
        return cls(
            SensorDataContainer([SensorDataItem.create(data, **meta_kwargs)]),
            logger=logger,
            force_iterative=force_iterative,
        )
