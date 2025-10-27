"""
This module contains utility functions to manipulate and condition arrays
before computation.
"""

from gpy.core._types import Arr64


def _condition_array(arr: Arr64) -> Arr64:
    """
    Function to condition 1D arrays into 2D column vectors and do nothing to
    arrays above 1D.

    One dimensional arrays with n samples have shape (n,) which can be
    problematic for the computations required for Gaussian Process Regression
    (GPR). Data used in regression is often expected in the form
    (n_samples, n_features) where each feature is represented as a separate
    dimension in the data matrix, and the rows represent the number of samples.
    Because of this, machine learning conventions prefer the more interpretable
    shape of (n, 1) to (n,) as it more clearly expresses the number of
    dimensions the data holds. Additionally, column vectors are much easier to
    work with during matrix arithmatic, which is a large part of any regression
    program.

    Args:
        arr (Arr64): Array to be conditioned.

    Returns:
        Arr64: Conditioned array.

    """

    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _condition_arrays(arr1: Arr64, arr2: Arr64) -> tuple[Arr64, Arr64]:
    """
    This function works identically to "_condition_array" but has been
    modified to condition pairs of arrays which is often necessary in the
    program (eg. conditioning x, x' before computing K(x, x')).

    Args:
        arr1 (Arr64): First array to be conditioned.
        arr2 (Arr64): Second array to be conditioned.

    Returns:
        tuple[Arr64, Arr64]: tuple of conditioned arrays.

    """
    return _condition_array(arr1), _condition_array(arr2)
