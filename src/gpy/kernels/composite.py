"""
Overview:
    This module implements composite kernels, which are constructed by
    combining two or more base kernels through addition or multiplication.
    Composite kernels allow Gaussian Process models to represent more complex
    covariance structures by blending different kinds of smoothness,
    periodicity, or scaling behaviors.

Usage:
    Use composite kernels to model functions with multiple interacting
    properties — for example, smooth long-term trends (RBF) with superimposed
    periodic variations (Periodic), or independent additive effects of
    different features.

Mathematical Properties:
    Additive Combination:
        K(x, x') = K₁(x, x') + K₂(x, x')
        Represents a sum of independent covariance sources.
        The resulting GP is the sum of two independent Gaussian Processes.

    Multiplicative Combination:
        K(x, x') = K₁(x, x') * K₂(x, x')
        Represents interaction between structures.
        For example, RBF * Periodic models a smoothly varying periodic function.

    Symmetry:
        Composite kernels remain symmetric if all component kernels are
        symmetric.

    Expressiveness:
        Composite kernels inherit the flexibility of their components and can
        approximate highly structured and nonstationary behaviors.
"""

from enum import Enum

import numpy as np

from gpy._utils._preprocess import _process_arrays
from gpy._utils._validation_utils import (
    _validate_kernel_tuple,
    _validate_set_params,
)
from gpy.core._base import Kernel
from gpy.core._types import Arr64, FloatSeq


class Operation(Enum):
    ADD = "ADD"
    MULTIPLY = "MULTIPLY"


class CompositeKernel(Kernel):
    @property
    def PARAM_ORDER(self) -> tuple[str, ...]:
        return tuple(
            param for kernel in self.kernels for param in kernel.PARAM_ORDER
        )

    @property
    def BOUNDS(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            bound for kernel in self.kernels for bound in kernel.BOUNDS
        )

    def __init__(self, kernels: tuple[Kernel, ...], operation: str) -> None:
        self.kernels = _validate_kernel_tuple(kernels)

        if not isinstance(operation, str):
            raise TypeError("Error: 'operation' argument expected as a string.")

        operation_list = [o.value for o in Operation]
        if operation.upper() not in operation_list:
            raise TypeError(
                "Error: Invalid operation entered. Valid operations include: "
                f"{operation_list} (case insensitive)"
            )

        self.operation = Operation(operation.upper())

    def compute(self, arr1: FloatSeq, arr2: FloatSeq) -> Arr64:
        arr1, arr2 = _process_arrays(
            arr1,
            arr2,
            "composite input 1",
            "composite input 2",
            allow_negative=True,
        )

        if self.operation == Operation.ADD:
            result = np.zeros((arr1.shape[0], arr2.shape[0]))
            for kernel in self.kernels:
                result += kernel.compute(arr1, arr2)
        elif self.operation == Operation.MULTIPLY:
            result = np.ones((arr1.shape[0], arr2.shape[0]))
            for kernel in self.kernels:
                result *= kernel.compute(arr1, arr2)
        else:
            raise TypeError(f"Error: Unsupported operation '{self.operation}'.")

        return result

    def get_params(self) -> list[float]:
        return [
            float(param)
            for kernel in self.kernels
            for param in kernel.get_params()
        ]

    def set_params(self, params: FloatSeq) -> None:
        expected_length = sum(
            len(kernel.PARAM_ORDER) for kernel in self.kernels
        )

        params = _validate_set_params(
            params,
            "Composite Kernel Hyperparameters",
            expected_length=expected_length,
        )

        # distribute params to respective kernels
        idx = 0
        for kernel in self.kernels:
            n_params = len(kernel.PARAM_ORDER)
            kernel.set_params(params[idx : idx + n_params])
            idx += n_params
