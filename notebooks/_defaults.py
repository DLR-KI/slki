# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#

# Jupyter Notebook defaults
# =========================
#
# This "notebook" defines the common properties for all notebooks as well as loads the chosen dataset.
#

from collections.abc import Callable
from functools import partial
import hashlib
import math
import os
import pickle
import random
import re
from typing import Any, overload

from dlr.ki.logging import load_default
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from slki.config import Config, TNorm
from slki.data import SensorData
from slki.data.data_import import load, load_all_typed
from slki.utils.debug import ensure_deterministic
from slki.viz.utils import clear_axes


# User Settings #######################################################################################################

SEED = 42
CACHING_ENABLED = True

dataset_file_directory = "/home/lab/slki/Dataset/preprecessed/v4/Oct24-data"

raw_data: bool = False  # raw data vs. processed data
resample_size: int | None = 1000  # resample the data to N points per signal
normalize: TNorm | None = "mone_one_zero_fix"  # normalize the data to [-1, 1], but keep the zero point fixed at 0.0
# Hint: raw and not resampled data requires a lot of memory!

dataset_type = "points-kionix-sh-z-" + ("raw" if raw_data else "dt")
datasets_filenames = [
    f"HighSpeedCooridore_Oct24-{dataset_type}%(chunk?).pkl",
    f"Güter_Oct24-{dataset_type}%(chunk?).pkl",
    f"Regio_Oct24-{dataset_type}%(chunk?).pkl",
]
datasets_labels = ["Fernverkehr", "Güterverkehr", "Regioverkehr"]
del dataset_type

_signal_memory: list[dict[int, tuple[np.ndarray, dict[str, Any]]]] = [{} for _ in range(len(datasets_filenames))]


# Initialize Notebook #################################################################################################


def init_notebook():
    ensure_deterministic(SEED)
    load_default("../logs/notebooks.log")
    os.makedirs("cache", exist_ok=True)


def get_data_hash_key(*data: str) -> str:
    data_hash = "\n".join(datasets_filenames + [str(resample_size), str(normalize)] + list(data))

    h = hashlib.new("sha256")
    h.update(data_hash.encode())
    return h.hexdigest()[:8]


# Load data ###########################################################################################################


def load_datasets_chunk_lengths() -> list[list[int]]:
    # Load dataset meta data to get their chunk lengths.
    Config.LOAD_METADATA_ONLY = True
    dataset_lengths = [
        [len(meta) for _, meta in load(os.path.join(dataset_file_directory, filename))]
        for filename in datasets_filenames
    ]
    Config.LOAD_METADATA_ONLY = False
    return dataset_lengths


def ballance_data(
    datasets: list[SensorData],
    max_length: int | None = None,
) -> tuple[list[SensorData], tuple[np.ndarray, list[dict[str, Any]], list[int]]]:
    # Search for the smallest dataset and randomly takes out the same number of samples out of the other datasets.
    # In the end all different datasets (different labels) will habe the same amount of data.

    def pick_random(sensor_data: SensorData, n: int) -> SensorData:
        if len(sensor_data) <= n:
            return sensor_data

        generator = np.random.default_rng(SEED)
        indices = generator.choice(len(sensor_data), n, replace=False)
        sensor_data.data = sensor_data.data[indices]
        sensor_data.meta = [sensor_data.meta[idx] for idx in indices]

        return sensor_data

    max_length = max_length or min([len(dataset) for dataset in datasets])
    datasets = [pick_random(dataset, max_length) for dataset in datasets]

    # Combine all datasets to a single larger but ballanced dataset and shuffle the data.
    data = np.vstack([dataset.data for dataset in datasets])
    meta = sum([dataset.meta for dataset in datasets], [])
    targets = sum([[idx] * max_length for idx in range(len(datasets_labels))], [])
    assert len(data) == len(meta) == len(targets)

    indices = list(range(len(data)))
    random.shuffle(indices)

    dataset = data[indices]
    metadata = [meta[idx] for idx in indices]
    targets = [targets[idx] for idx in indices]

    return datasets, (dataset, metadata, targets)


def load_datasets() -> list[SensorData]:
    # Prepare dataset caching.
    data_hash = get_data_hash_key()
    cache_file = f"./cache/datasets-{data_hash}.pkl"

    # Load data.
    if CACHING_ENABLED and os.path.exists(cache_file):
        print(f"Load previous cached dataset from '{cache_file}' file.")
        with open(cache_file, "rb") as f:
            datasets = pickle.load(f)
    else:
        datasets = [
            load_all_typed(
                os.path.join(dataset_file_directory, filename),
                resample_size=resample_size,
                normalize=normalize,
            )
            for filename in datasets_filenames
        ]
        if CACHING_ENABLED:
            with open(cache_file, "wb") as f:
                pickle.dump(datasets, f)

    return datasets


def load_ballance_datasets(
    max_length: int | None = None,
) -> tuple[list[SensorData], tuple[np.ndarray, list[dict[str, Any]], list[int]]]:
    datasets = load_datasets()
    return ballance_data(datasets, max_length=max_length)


def load_signal(
    dataset_index: int,
    signal_index: int,
    dataset_lengths: list[list[int]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if CACHING_ENABLED and _signal_memory[dataset_index].__contains__(signal_index):
        return _signal_memory[dataset_index][signal_index]

    filepath, chunk_signal_index = _get_dataset_file_path(dataset_index, signal_index, dataset_lengths)
    sensor_data = load_all_typed(filepath, resample_size=resample_size, normalize=normalize)

    if not CACHING_ENABLED:
        return sensor_data.data[chunk_signal_index], sensor_data.meta[chunk_signal_index]

    start_index = signal_index - chunk_signal_index
    for idx, (data, meta) in enumerate(sensor_data):
        signal = data[: meta["sample_length"]]
        _signal_memory[dataset_index][start_index + idx] = (signal, meta)

    return _signal_memory[dataset_index][signal_index]


def _get_dataset_file_path(
    dataset_index: int,
    signal_index: int,
    dataset_lengths: list[list[int]] | None = None,
) -> tuple[str, int]:
    dataset_lengths = dataset_lengths or load_datasets_chunk_lengths()
    chunk_num = 1
    chunk_signal_index = signal_index
    while chunk_signal_index + 1 >= dataset_lengths[dataset_index][chunk_num - 1]:
        chunk_signal_index -= dataset_lengths[dataset_index][chunk_num - 1]
        chunk_num += 1

    filename = re.escape(datasets_filenames[dataset_index]).replace("%\\(chunk\\?\\)", f"(_?0*{chunk_num})?")
    filename = [f for f in os.listdir(dataset_file_directory) if re.match(filename, f)][0]
    filepath = os.path.join(dataset_file_directory, filename)

    return filepath, chunk_signal_index


# Print data table #####################################################################################################


@overload
def print_db_length_table(datasets_or_targets: list[SensorData]) -> None: ...


@overload
def print_db_length_table(datasets_or_targets: list[int]) -> None: ...


def print_db_length_table(datasets_or_targets: list[SensorData] | list[int]) -> None:
    assert datasets_or_targets, "No data."

    if isinstance(datasets_or_targets[0], SensorData):
        # datasets_or_targets is list[SensorData]
        iterator = (len(dataset) for dataset in datasets_or_targets)  # type: ignore[arg-type]
    else:
        # datasets_or_targets is list[int]
        _, iterator = np.unique(datasets_or_targets, return_counts=True)  # type: ignore[call-overload]

    print(tabulate(zip(datasets_labels, iterator, strict=False), headers=["Dataset", "Length"]))


# Visualization helpers ###############################################################################################


def plot_clusters(
    clusters,
    dataset,
    n_cols: int = 3,
    avg_fn: Callable | None = None,
) -> None:
    avg_fn = avg_fn or partial(np.average, axis=0)
    # if your cluster numbers are not 0-based or continuous
    cluster_dict = dict(zip(set(clusters), range(len(set(clusters))), strict=False))
    _n_clusters = len(set(clusters))
    n_cols = min(n_cols, _n_clusters)
    n_rows = math.ceil(_n_clusters / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 3 * n_rows))
    axes_flattened = axes.flatten()
    clear_axes(axes_flattened[_n_clusters:])
    for cluster_num in cluster_dict.values():
        cluster = []
        for i in range(len(clusters)):
            if cluster_dict[clusters[i]] == cluster_num:
                axes_flattened[cluster_num].plot(dataset[i], c="gray", alpha=0.4)
                cluster.append(dataset[i])
        if len(cluster) > 0:
            axes_flattened[cluster_num].plot(avg_fn(np.vstack(cluster)), c="red")
        axes_flattened[cluster_num].set_title(f"Cluster {cluster_num}")
        axes_flattened[cluster_num].text(
            0.99,
            1.1,
            f"{len(cluster)} sample(s)",
            ha="right",
            va="top",
            transform=axes_flattened[cluster_num].transAxes,
        )

    fig.suptitle("Clusters", fontsize=20, y=1.0)
    fig.tight_layout()
    plt.show()
