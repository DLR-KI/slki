# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""This module provides functions for detecting and clustering peaks in a signal."""

from logging import Logger

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_prominences
from sklearn.cluster import DBSCAN

from ..config import Config


class PeaksClusteringResult:
    """Represents the result of peak clustering."""

    def __init__(self, clustering: DBSCAN):
        """
        Initialize a peak clustering result object.

        Args:
            clustering (DBSCAN): The DBSCAN clustering object used to assign labels to each peak.
        """
        self.labels = clustering.labels_
        """The cluster labels assigned to each peak."""

        # Number of clusters in labels, ignoring noise if present.
        self.n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        """The number of clusters in the peak labels."""
        self.n_noise_ = list(self.labels).count(-1)
        """The number of noise points in the peak labels."""

    def plot_peaks(self, peaks: np.ndarray, ax: plt.Axes, delta: int = 10):
        """
        Plot the peaks on the given axes.

        Args:
            peaks (np.ndarray): Array of peak values.
            ax (plt.Axes): The axes on which to plot the peaks.
            delta (int, optional): Additional rectangle size to be drawn around the clusters. Default to 10.
        """
        unique_labels = set(self.labels)
        for label in unique_labels:
            if label == -1:
                continue  # skip noise points

            label_mask = self.labels == label
            x = peaks[label_mask]
            y = ax.get_ylim()
            ax.add_patch(
                Rectangle(
                    (min(x) - delta, min(y)),
                    max(x) - min(x) + 2 * delta,
                    max(y) - min(y),
                    linewidth=1,
                    edgecolor="none",
                    facecolor="b",
                    alpha=0.1,
                )
            )


def detect_upper_peaks(signal: np.ndarray, sample_length: int) -> np.ndarray:
    """
    Detects upper peaks in a given signal.

    Args:
        signal (np.ndarray): The input signal to analyze.
        sample_length (int): The length of the original signal.

    Returns:
        np.ndarray: An array containing the indices of the upper peaks.

    References:
        - <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>
    """
    peaks, _ = find_peaks(signal, distance=Config.peak_distance(len(signal), sample_length))
    prominences, *_ = peak_prominences(signal, peaks, wlen=Config.peak_prominence_wlen(len(signal), sample_length))
    peak_minimum = max(0.05, Config.PEAK_MINIMUM_SIGNAL_FACTOR * np.mean(np.abs(signal[peaks])))
    return np.array(
        [
            peak
            for peak, prominence in zip(peaks, prominences, strict=False)
            if prominence > Config.MINIMAL_PEAK_PROMINENCE and np.abs(signal[peak]) > peak_minimum
        ]
    )


def detect_lower_peaks(signal: np.ndarray, sample_length: int) -> np.ndarray:
    """
    Detects lower peaks in a given signal.

    Args:
        signal (np.ndarray): The input signal to analyze.
        sample_length (int): The length of the original signal.

    Returns:
        np.ndarray: An array containing the indices of the lower peaks.
    """
    return detect_upper_peaks(-signal, sample_length)


def detect_peaks(signal: np.ndarray, sample_length: int) -> np.ndarray:
    """
    Detects both upper and lower peaks in a given signal.

    Args:
        signal (np.ndarray): The input signal to analyze.
        sample_length (int): The length of the original signal.

    Returns:
        np.ndarray: A numpy array of unique peak indices.
    """
    upper_peaks = detect_upper_peaks(signal, sample_length)
    lower_peaks = detect_lower_peaks(signal, sample_length)
    return np.unique(np.sort(np.hstack((upper_peaks, lower_peaks))))


def cluster_peaks(peaks: np.ndarray, signal_length: int, sample_length: int) -> PeaksClusteringResult:
    """
    Clusters peaks found in a signal using DBSCAN.

    Args:
        peaks (np.ndarray): The indices of the peak values to be clustered.
        signal_length (int): The length of the current signal, e.g. after resampling etc.
        sample_length (int): The length of the original signal.

    Returns:
        PeaksClusteringResult: An object containing the clustering result.
    """
    eps = Config.cluster_max_sample_distance(signal_length, sample_length)
    clustering = DBSCAN(eps=eps, min_samples=2).fit(peaks.reshape(-1, 1))
    return PeaksClusteringResult(clustering)


def detect_and_cluster_and_plot_peaks(
    signal: np.ndarray,
    sample_length: int,
    ax: plt.Axes,
    logger: Logger | None = None,
    print_console: bool = False,
) -> tuple[PeaksClusteringResult, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detects peaks, clusters them using DBSCAN, and plots the result.

    Args:
        signal (np.ndarray): The input signal to analyze.
        sample_length (int): The length of the original signal.
        ax (plt.Axes): The axes object to plot on.
        logger (Optional[Logger], optional): An optional logger instance. Defaults to None.
        print_console (bool, optional): Whether to print information to console. Defaults to False.
    """
    # detect and plot peaks
    upper_peaks = detect_upper_peaks(signal, sample_length)
    if upper_peaks.any():
        ax.scatter(upper_peaks, signal[upper_peaks], marker="o", color="tab:purple", s=40, facecolors="none")
    lower_peaks = detect_lower_peaks(signal, sample_length)
    if lower_peaks.any():
        ax.scatter(lower_peaks, signal[lower_peaks], marker="o", color="tab:cyan", s=40, facecolors="none")
    peaks = np.unique(np.sort(np.hstack((upper_peaks, lower_peaks))))
    if logger:
        logger.info(f"Estimated number of peaks: {len(peaks)}")
    if print_console:
        print(f"Estimated number of peaks: {len(peaks)}")

    # detect clusters
    if peaks.any():
        clustering_results = cluster_peaks(peaks, len(signal), sample_length)
        if logger:
            logger.info(f"Estimated number of clusters: {clustering_results.n_clusters_}")
            logger.info(f"Estimated number of noise points: {clustering_results.n_noise_}")
        if print_console:
            print(f"Estimated number of clusters: {clustering_results.n_clusters_}")
            print(f"Estimated number of noise points: {clustering_results.n_noise_}")
        clustering_results.plot_peaks(peaks, ax, Config.cluster_rect_drawing_delta(len(signal), sample_length))

    return clustering_results, lower_peaks, upper_peaks, peaks
