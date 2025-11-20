# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""This module provides utility functions for visualization in matplotlib."""

from collections.abc import Generator, Sequence
from contextlib import contextmanager
import logging
from typing import Any

import matplotlib.pyplot as plt


def clear_axis(ax: plt.Axes) -> None:
    """
    Remove ticks and disable frame and axis rendering.

    Args:
        ax (plt.Axes): Single axis to clear.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.patch.set_visible(False)


def clear_axes(axes: list[plt.Axes]) -> None:
    """
    Remove ticks and disable frame and axis rendering.

    Args:
        axes (List[plt.Axes]): List of axis to clear.
    """
    for ax in axes:
        clear_axis(ax)


def plot_grid_base(  # noqa: PLR0913
    rows: int,
    cols: int,
    *,
    factor: float = 1.0,
    wspace: float = 0.1,
    hspace: float = 0.1,
    clear_axes: bool = True,
    **subplots_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create figure and axes with default grid settings.

    Args:
        rows (int): Number of rows of the subplot grid.
        cols (int): Number of columns of the subplot grid.
        factor (float, optional): Figure size factor. Figure size will be `(cols*factor, rows*factor)`. Defaults to 1.
        wspace (float, optional): The amount of width reserved for space between subplots, expressed as a fraction of
            the average axis width. Defaults to 0.1.
        hspace (float, optional): The amount of height reserved for space between subplots, expressed as a fraction of
            the average axis height. Defaults to 0.1.
        clear_axes (bool, optional): Remove ticks and disable frame and axis rendering. Defaults to True.
        subplots_kwargs (Dict[str, Any], optional): Additional `plt.subplots` arguments. Defaults to {}.

    Returns:
        Tuple[plt.Figure, plt.Axes]: figure, axes (non squeezed by default)
    """
    gs = {"wspace": wspace, "hspace": hspace}
    gs.update(subplots_kwargs.pop("gridspec_kw", {}))
    kwargs = {
        "ncols": cols,
        "nrows": rows,
        "gridspec_kw": gs,
        "squeeze": False,  # subplots args
        "figsize": (cols * factor, rows * factor),
        "tight_layout": True,  # figure args
    }
    kwargs.update(subplots_kwargs)
    fig, axes = plt.subplots(**kwargs)  # type: ignore[call-overload]
    if clear_axes:
        for ax in axes.flat:
            clear_axis(ax)
    return fig, axes


def flat_axes(axes: plt.Axes | Sequence[plt.Axes] | Sequence[Sequence[Any]]) -> list[plt.Axes]:
    """
    Flatten a nested list of axes into a single list.

    Args:
        axes (matplotlib.axes.Axes or list or tuple): The axes to flatten.

    Returns:
        List[matplotlib.axes.Axes]: The flattened list of axes.
    """
    axes_flatten = []
    if hasattr(axes, "ravel"):
        axes_flatten = axes.ravel().tolist()
    elif isinstance(axes, Sequence):
        for ax in axes:
            axes_flatten += flat_axes(ax)
    else:
        axes_flatten = [axes]
    return axes_flatten


@contextmanager
def latex_backend(*, enable: bool = True) -> Generator[None, None, None]:
    """
    A context manager that enables LaTeX backend for matplotlib.

    The LaTeX backend is enabled by updating the `plt.rcParams` dictionary with the necessary settings for using LaTeX
    with matplotlib.
    The original `rc_params` are saved and restored after the code inside the context manager is executed.

    When the `enable` parameter is set to False, the context manager does nothing and the code inside is not affected.

    Args:
        enable (bool, optional): Whether to enable the LaTeX backend. Default is True.

    Example:
        ```python
        with latex_backend():
            # Code that requires LaTeX backend
            ...
        ```
    """
    if not enable:
        yield
        return

    rc_params = plt.rcParams.copy()
    try:
        plt.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
            }
        )
        yield
    finally:
        plt.rcParams.update(rc_params)


def savefig(
    filepath_without_ext: str,
    fig: plt.Figure | None = None,
    exts: list[str] | None = None,
    latex_backend_exts: list[str] | None = None,
    logger: logging.Logger | None = None,
    **kwargs,
) -> None:
    """
    Save a matplotlib figure.

    This methods supports multiple file formats at once including LaTeX backend support.

    Args:
        filepath_without_ext (str): The filepath without extension to save the figure.
        fig (Optional[plt.Figure], optional): The matplotlib figure to save. If not provided, the current figure will be used.
        exts (List[str], optional): The list of file extensions to save the figure in. Default is ["png", "svg", "pdf", "pgf"].
        latex_backend_exts (List[str], optional): The list of file extensions that require LaTeX backend. Default is ["pgf"].
        logger (Optional[logging.Logger], optional): The logger to use for logging. If not provided, the default logger will be used.
        **kwargs: Additional keyword arguments to pass to the `savefig` function of matplotlib.
    """  # noqa: E501
    exts = exts or ["png", "svg", "pdf", "pgf"]
    latex_backend_exts = latex_backend_exts or ["pgf"]
    fig = fig or plt.gcf()
    kwargs = kwargs or {"bbox_inches": "tight"}
    for ext in exts:
        enable_latex = ext in latex_backend_exts
        with latex_backend(enable=enable_latex):
            fig.savefig(f"{filepath_without_ext}.{ext}", **kwargs)
            (logger or logging).info(f"Saved: {filepath_without_ext}.{ext}")
