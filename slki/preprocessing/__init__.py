# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
from .absolute import Absolute
from .boost import Boost
from .denoise import Denoise
from .detect import DetectSignal
from .double_integrate import DoubleIntegrate
from .normalize import Normalize
from .outlier import Outlier
from .outlier_elemination import OutlierElimination
from .resample import Resample
from .smooth import Smooth
from .stage import Stage


STAGES: dict[str, type[Stage]] = {
    "abs": Absolute,
    "boost": Boost,
    "denoise": Denoise,
    "detect": DetectSignal,
    "double_integration": DoubleIntegrate,
    "norm": Normalize,
    "outlier": Outlier,
    "outlier2": OutlierElimination,
    "resample": Resample,
    "smooth": Smooth,
}
"""A dictionary mapping stage names to their respective stage classes."""


def get_stage_cls(name: str) -> type[Stage]:
    """
    Retrieves a stage class by its name.

    Args:
        name (str): The unique name of the desired stage.

    Raises:
        ValueError: If the specified stage is not found in the predefined stages.

    Returns:
        Type[Stage]: The stage class corresponding to the given name.
    """
    if name not in STAGES:
        raise ValueError(f"Stage '{name}' is not defined.")

    return STAGES[name]


__all__ = [
    "STAGES",
    "Absolute",
    "Boost",
    "Denoise",
    "DetectSignal",
    "DoubleIntegrate",
    "Normalize",
    "Outlier",
    "OutlierElimination",
    "Resample",
    "Smooth",
    "Stage",
    "get_stage_cls",
]
