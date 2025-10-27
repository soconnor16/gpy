"""
This module is used to validate and preprocess data before it is used in actual
computation. This is necessary to provide consistent usage, and more helpful
error messages.
"""

import numpy as np

from gpy.core._base import Kernel
from gpy.core._types import Arr64, FloatSeq, Numeric


# ---------------------------- Kernel Validation ---------------------------- #
def _validate_set_params(
    params: FloatSeq,
    name: str,
    expected_length: int,
) -> Arr64:
    params_array = _validate_array(params, name, False)
    params_array = params_array.flatten()

    if len(params_array) != expected_length:
        raise ValueError(
            f"Error: '{name}' expects {expected_length} hyperparameters."
            f"Got: {len(params_array)}.",
        )

    return params_array


def _validate_kernel_tuple(kernels: tuple[Kernel, ...]) -> tuple[Kernel, ...]:
    # make sure 'kernels' is a tuple
    # try converting to accomodate for lists, sets, etc.
    try:
        kernels = tuple(kernels)

    except TypeError as e:
        raise TypeError(
            f"{e!s}"
            "Error: 'kernels' input is malformed and could not be "
            "converted to a tuple. Please ensure your 'kernels' argument for "
            "a CompositeKernel instance is a tuple of kernels"
        )

    if not all(isinstance(k, Kernel) for k in kernels):
        raise TypeError(
            "Error: All elements of the 'kernels' argument must be valid kernel"
            " instances"
        )

    if len(kernels) < 2:
        raise ValueError("Error: 'kernels' should have at least two kernels")

    return kernels


# --------------------------- General Validation ---------------------------- #


def _validate_array(
    array: FloatSeq,
    name: str,
    allow_negative: bool,
) -> Arr64:
    """
    Function to validate numeric arrays and array-like objects.

    This function is used to preprocess data before being used for computation.
    It attempts allow for more flexible usage of the program by attempting to
    convert inputs into consistent NumPy arrays rather than only accepting
    specific input types, leveraging NumPy's efficiency while permitting the use
    of more ergonomic Python native variable types.

    Args:
        array (FloatSeq): Array to be validated and preprocessed before used in
                          computation.
        name (str): "name" representing the arrays use for more descriptive
                    error messages.
        allow_negative (bool): Whether the array should allow negative values.

    Raises:
        ValueError: If the array conversion fails, the array is empty, the
                    array contains 'inf' or 'nan' values, or if the array
                    contains negative values when it should not.

    Returns:
        Arr64: Validated and preprocessed array ready for calculations.

    """

    try:
        array = np.array(array, dtype=np.float64)

    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"Error processing '{name}': {exc!s}") from exc

    if not array.size:
        raise ValueError(f"Error: '{name}' cannot be an empty array.")

    if np.any(np.isinf(array)) or np.any(np.isnan(array)):
        if np.any(np.isinf(array)):
            raise ValueError(f"Error: '{name}' cannot contain 'inf' values.")
        if np.any(np.isnan(array)):
            raise ValueError(f"Error: '{name}' cannot contain 'nan' values.")
    if not allow_negative and np.any(array <= 0):
        raise ValueError(
            f"Error: '{name}' must contain only positive, non-zero values.",
        )

    return array.astype(np.float64)


def _validate_numeric_value(
    value: Numeric,
    name: str,
    allow_negative: bool,
) -> np.float64:
    """
    Function to validate individual numeric values.

    This function is used to preprocess individual numeric values before they
    are used in computation. It is mostly used for the initialization of
    Kernels to ensure that the inputted hyperparameters are valid.

    Args:
        value (Numeric): Numeric value to be validated and processed.
        name (str): "name" representing the values use for more descriptive
                    error messages.
        allow_negative (bool): Whether the value should be allowed to be
                               negative.

    """

    try:
        value = np.float64(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"Error processing '{name}': {exc!s}") from exc

    if np.isnan(value):
        raise ValueError(f"Error: '{name}' cannot be 'nan'.")
    if np.isinf(value):
        raise ValueError(f"Error: '{name}' cannot be 'inf'.")

    if not allow_negative and value < 0:
        raise ValueError(
            f"Error: '{name}' must contain only positive, non-zero values.",
        )

    return np.float64(value)
