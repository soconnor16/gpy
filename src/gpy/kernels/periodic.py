"""
Overview:
    This module implements the periodic kernel. The periodic kernel is designed
    to model functions that repeat at regular intervals. It is particularly
    effective for capturing patterns such as seasonal effects, cyclic
    behaviors, or any other repeating phenomena in time series or spatial data.

Usage:
    Use of the periodic kernel is best when the underlying process exhibits
    regular oscillations or repeating patterns. It can also be combined with
    other kernels (e.g., RBF) to capture both periodicity and smooth trends.

Mathematical Properties:
    Kernel Function:
        K(x, x') = exp(-2 * sin^2(π * ||x - x'|| / p) / l^2)

        Where:
            l – length scale, controlling the smoothness of the periodic pattern
            p – period, determining the distance between repetitions

    Symmetry:
        K(x, x') = K(x', x)

    Periodicity:
        K(x, x' + p) = K(x, x')

    Interpretability:
        The kernel enforces strict periodic behavior while allowing smooth
        variations within each period through the length scale parameter.
"""

import numpy as np

from gpy._utils._computation_utils import compute_euclidean_distance
from gpy._utils._preprocess import _process_arrays
from gpy._utils._validation_utils import (
    _validate_numeric_value,
    _validate_set_params,
)
from gpy.core._base import Kernel
from gpy.core._types import Arr64, FloatSeq


class PeriodicKernel(Kernel):
    @property
    def PARAM_ORDER(self) -> tuple[str, ...]:
        return ("sigma", "length_scale", "period")

    @property
    def BOUNDS(self) -> tuple[tuple[float, float], ...]:
        return ((1e-2, 5e2), (1e-2, 5e1), (1e-2, 1e1))

    def __init__(
        self, sigma: float, length_scale: float, period: float
    ) -> None:
        self.sigma = _validate_numeric_value(
            sigma,
            "Periodic Sigma",
            False,
        )

        self.length_scale = _validate_numeric_value(
            length_scale,
            "Periodic Length Scale",
            False,
        )
        self.period = _validate_numeric_value(
            period,
            "Periodic Period",
            False,
        )

    def compute(self, arr1: FloatSeq, arr2: FloatSeq) -> Arr64:
        arr1, arr2 = _process_arrays(
            arr1,
            arr2,
            "Periodic input 1",
            "Periodic input 2",
            True,
        )

        abs_dist = compute_euclidean_distance(arr1, arr2)

        sin_comp = np.sin((np.pi * abs_dist) / self.period)

        return self.sigma**2 * np.exp(-2 * sin_comp**2 / self.length_scale**2)

    def get_params(self) -> list[float]:
        return [float(self.sigma), float(self.length_scale), float(self.period)]

    def set_params(self, params: FloatSeq) -> None:
        params = _validate_set_params(
            params,
            "Periodic Kernel Hyperparameters",
            expected_length=len(self.PARAM_ORDER),
        )

        self.sigma = params[0]
        self.length_scale = params[1]
        self.period = params[2]
