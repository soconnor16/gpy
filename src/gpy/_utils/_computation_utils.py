"""
This module holds utility functions for common array calculations needed for
the program.
"""

import numpy as np
from scipy.spatial import distance

from gpy.core._types import Arr64


def compute_euclidean_distance(arr1: Arr64, arr2: Arr64) -> Arr64:
    """
    Function to compute the euclidean distance matrix between two input arrays.

    Args:
        arr1 (Arr64): First input array.
        arr2 (Arr64): Second input array.

    Returns:
        Arr64: Euclidean distance matrix between arr1 and arr2.

    """

    dist = distance.cdist(arr1, arr2, metric="euclidean")

    return dist.astype(np.float64)


def compute_square_euclidean_distance(arr1: Arr64, arr2: Arr64) -> Arr64:
    """
    Function to compute the square euclidean distance matrix between two input
    arrays.

    Args:
        arr1 (Arr64): First input array.
        arr2 (Arr64): Second input array.

    Returns:
        Arr64: Square euclidean distance matrix between arr1 and arr2.

    """

    sqdist = distance.cdist(arr1, arr2, metric="sqeuclidean")

    return sqdist.astype(np.float64)
