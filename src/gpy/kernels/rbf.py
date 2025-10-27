"""
Overview:
    This module implements the rbf kernel. The rbf kernel is one of the most
    popular kernel functions for machine learning applications due to its
    flexibility and ability to capture non-linear relationships within data.
    Rbf kernels also contain only one hyperparameter, making them simple and
    easy to tune.

Usage:
    When facing a new problem, the rbf kernel is often a good jumping off
    point while more specific kernels can be brought in where the rbf
    kernel fails.

Mathematical Properties:
    Kernel Function: K(x, x') = exp(-||x - x'||^2 / 2l^2)
        Where l is the kernel's length scale and determines the smoothness of
        the predicted kernel function.

    Symmetry: K(x, x') = K(x', x)

    Monotonicity: K decreases monotonically with distance (||x - x'||^2)

    Universal Approximation: The rbf kernel has the universal
                             approximation property. This means that given
                             sufficient data, it is able to accurately fit to
                             the predictive function to an arbitrary degree of
                             accuracy.
"""

import numpy as np

from gpy._utils._computation_utils import compute_square_euclidean_distance
from gpy._utils._preprocess import _process_arrays
from gpy._utils._validation_utils import (
    _validate_numeric_value,
    _validate_set_params,
)
from gpy.core._base import Kernel
from gpy.core._types import Arr64, FloatSeq


class RBFKernel(Kernel):
    @property
    def PARAM_ORDER(self) -> tuple[str, ...]:
        return ("sigma", "length_scale")

    @property
    def BOUNDS(self) -> tuple[tuple[float, float], ...]:
        return ((1e-2, 5e2), (1e-2, 5e1))

    def __init__(self, sigma: float, length_scale: float):
        self.sigma = _validate_numeric_value(
            sigma,
            "RBF Sigma",
            False,
        )

        self.length_scale = _validate_numeric_value(
            length_scale,
            "RBF Length Scale",
            False,
        )

    def compute(self, arr1: FloatSeq, arr2: FloatSeq) -> Arr64:
        arr1, arr2 = _process_arrays(
            arr1,
            arr2,
            "rbf input 1",
            "rbf input 2",
            True,
        )

        sqdist = compute_square_euclidean_distance(arr1, arr2)

        return self.sigma**2 * np.exp(-0.5 * sqdist / self.length_scale**2)

    def get_params(self) -> list[float]:
        return [float(self.sigma), float(self.length_scale)]

    def set_params(self, params):
        params = _validate_set_params(
            params,
            "RBF Kernel Hyperparameters",
            len(self.PARAM_ORDER),
        )

        self.sigma = params[0]
        self.length_scale = params[1]
