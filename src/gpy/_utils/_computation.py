from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg
from scipy.spatial import distance

from gpy._utils._constants import SMALL_EPSILON
from gpy._utils._types import Arrf64, f64

if TYPE_CHECKING:
    from gpy.GaussianProcess.gaussian_process import GaussianProcess


def compute_square_euclidean_distance(x1: Arrf64, x2: Arrf64) -> Arrf64:
    """
    Helper function which computes the square euclidean distance (x - x_i)^2
    between two input arrays.

    Args:
        - x1 (Arrf64): Input array 1 of shape
        - x2 (Arrf64): Input array 2 of shape

    Returns:
        Arrf64: The square euclidean distance matrix between input arrays.
    """
    square_dist = distance.cdist(x1, x2, metric="sqeuclidean")

    return square_dist.astype(np.float64)


def compute_absolute_distance(x1: Arrf64, x2: Arrf64) -> Arrf64:
    """
    Helper function which computes the absolute distance |x - x_i| between
    two input arrays.

    Args:
        - x1 (Arrf64): Input array 1
        - x2 (Arrf64): Input array 2

    Returns:
        Arrf64: The absolute distance tensor of shape (n, m, d) where
            result[i, j, k] = |x1[i, k] - x2[j, k]|
    """
    x1_expanded = x1[:, np.newaxis, :]
    x2_expanded = x2[np.newaxis, :, :]

    return np.abs(x1_expanded - x2_expanded)


def compute_lower_cholesky_decomposition(
    K: Arrf64, noise: float, max_attempts: int
) -> tuple[Arrf64, float]:
    """
    Computes the lower triangular Cholesky decomposition of a kernel matrix
    with adaptive noise regularization for numerical stability.

    Regularization is attempted in three ways:
    1. Decompose with current noise level
    2. Scale noise based on kernel matrix diagonal mean
    3. Add jitter based on negative eigenvalues if needed

    Args:
        - K (Arrf64): Kernel matrix of shape (n, n)
        - noise (float): Initial noise/jitter level to add to diagonal
        - max_attempts (int): Maximum number of decomposition attempts

    Returns:
        tuple[Arrf64, float]: Lower triangular Cholesky factor L and the
            final noise level used, where K + noise * I = L @ L.T

    Raises:
        ValueError: If decomposition fails after all strategies are exhausted
                    or if matrix has positive eigenvalues but still fails to
                    decompose.
    """
    n = K.shape[0]
    K_noise = np.empty_like(K)  # pre-allocate to avoid repeated allocations

    for attempt in range(max_attempts):
        # first strategy: decompose with current noise
        try:
            np.copyto(K_noise, K)
            K_noise.flat[:: n + 1] += noise
            lower_chol = linalg.cholesky(K_noise, lower=True)
            break
        except linalg.LinAlgError:
            # second strategy: add noise based on kernel matrix diagonal
            if attempt < max_attempts - 1:
                k_scale = float(np.mean(np.diag(K)))
                noise = max(10 * noise, k_scale * SMALL_EPSILON)

            # final strategy: add noise based on kernel matrix eigenvalues
            else:
                try:
                    np.copyto(K_noise, K)
                    K_noise.flat[:: n + 1] += noise
                    eigenvalues = linalg.eigvalsh(K_noise)
                    min_eig = np.min(eigenvalues)

                    if min_eig < 0.0:
                        jitter = np.abs(min_eig) + SMALL_EPSILON
                        K_noise.flat[:: n + 1] += jitter
                        lower_chol = linalg.cholesky(K_noise, lower=True)

                    else:
                        err_msg = (
                            "Error: Cholesky Decomposition of the Kernel "
                            "matrix failed despite positive eigenvalues."
                        )
                        raise ValueError(err_msg)

                except linalg.LinAlgError as exc:
                    err_msg = (
                        "Error: Numerical instability during Kernel matrix "
                        f"decomposition: {exc!s}"
                    )
                    raise ValueError(err_msg) from exc

    return lower_chol, noise


def compute_rmse_across_dataset(
    gp: "GaussianProcess", x_full: Arrf64, y_full: Arrf64
) -> f64:
    """
    Computes the root mean squared error (RMSE) of Gaussian process predictions
    across a dataset.

    Args:
        - gp (GaussianProcess): Fitted Gaussian process model
        - x_full (Arrf64): Input features of shape (n, d)
        - y_full (Arrf64): True target values of shape (n,)

    Returns:
        f64: RMSE = sqrt(mean((y_pred - y_true)^2))
    """
    y_pred = gp.predict(x_full)

    return np.float64(np.sqrt(np.mean((y_pred - y_full) ** 2)))
