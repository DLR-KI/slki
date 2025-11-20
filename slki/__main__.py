#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""This module contains the main entry point for the application and can be called via `python -m slki`."""

from logging import getLogger

from .config import Config
from .data.data_export import dump
from .pipeline import Pipeline
from .utils.debug import ensure_deterministic


def run() -> None:
    """
    Runs the main pipeline.

    This function ensures deterministic behavior (if desired), logs the configuration, and runs the defined pipeline.
    After running the pipeline, the processed data will be saved on the filesystem.
    """
    ensure_deterministic(Config.SEED)

    # log configuration
    getLogger("slki").debug(f"Config: {Config.__dict__}")

    # run pipeline
    pipeline = Pipeline()
    for idx, _ in enumerate(pipeline.import_data()):
        # run pipeline (on current data chunk)
        pipeline.run()

        # save processed data
        num_chunks = pipeline.get_number_of_chunks_estimation()
        if num_chunks is not None and num_chunks > 1:
            # save chunked data
            dump(pipeline.get_data_container(), chunk_number=idx + 1, chunk_digits=len(str(num_chunks)))
        else:
            # save data container (without "onload" chunking)
            dump(pipeline.get_data_container())


def main() -> None:
    """Main entry function to `slki` application."""
    from dlr.ki.logging import load_default

    load_default("logs/pipeline.log")

    try:
        run()
    except Exception as e:
        logger = getLogger("slki")
        logger.fatal("Unexpected error occurred.")
        logger.exception(e)


if __name__ == "__main__":
    main()
