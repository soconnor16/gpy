"""
Overview:
    This module implements the constant kernel. The constant kernel assigns a
    fixed covariance value between all pairs of points, regardless of their
    distance. It represents a uniform prior correlation structure and is often
    used to model global bias or scaling in Gaussian Process models.

Usage:
    Use the constant kernel when you expect a constant baseline correlation or
    when combining kernels to scale their contribution (e.g., multiplying with
    RBF or periodic kernels).

Mathematical Properties:
    Kernel Function:
        K(x, x') = c

        Where:
            c – the constant value, determining the overall variance magnitude.

    Symmetry:
        K(x, x') = K(x', x)

    Stationarity:
        The constant kernel is stationary, since it depends on neither x nor x'.

    Role in Composite Kernels:
        Acts as a scaling or bias term.
        For example, K_total = c * K_rbf(x, x') adds a variance scaling factor.

    Limiting Behavior:
        If c = 0, all points are uncorrelated.
        If c > 0, the covariance matrix becomes a constant matrix of value c.
"""

import numpy as np

from gpy._utils._preprocess import _process_arrays
from gpy._utils._validation_utils import (
    _validate_numeric_value,
    _validate_set_params,
)
from gpy.core._base import Kernel
from gpy.core._types import Arr64, FloatSeq


class ConstantKernel(Kernel):
    @property
    def PARAM_ORDER(self) -> tuple[str, ...]:
        return ("c",)

    @property
    def BOUNDS(self) -> tuple[tuple[float, float], ...]:
        return ((1e-2, 5e1),)

    def __init__(self, c: float) -> None:
        self.c = _validate_numeric_value(
            c, "Constant Kernel constant value", True
        )

    def compute(self, arr1: FloatSeq, arr2: FloatSeq) -> Arr64:
        arr1, arr2 = _process_arrays(
            arr1,
            arr2,
            "constant input 1",
            "constant input 2",
            True,
        )

        return self.c * np.ones((arr1.shape[0], arr2.shape[0]))

    def get_params(self) -> list[float]:
        return [self.c]

    def set_params(self, params: FloatSeq) -> None:
        params = _validate_set_params(
            params,
            "Constant Kernel Hyperparameters",
            len(self.PARAM_ORDER),
        )

        self.c = params[0]
