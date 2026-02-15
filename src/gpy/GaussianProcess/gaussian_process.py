"""
Gaussian Process Regression (GPR) implementation for supervised learning with
uncertainty quantification.

Gaussian Processes define a distribution over functions f(x) ~ GP(m(x), k(x,x'))
where m(x) is the mean function (assumed zero here) and k(x,x') is the
covariance/kernel function.

Given training data (X, y), predictions at new points X* are computed as:

    Mean:     μ* = K(X*, X) @ α,  where α = K(X, X)^{-1} @ y
    Variance: σ²* = K(X*, X*) - K(X*, X) @ K(X, X)^{-1} @ K(X, X*)

The Cholesky decomposition L (where K = L @ L.T) is used for numerical
stability, allowing efficient computation of α via solving L @ L.T @ α = y.

Hyperparameters (kernel parameters and noise) can be optimized by maximizing
the log marginal likelihood:

    log p(y|X,θ) = -0.5 * y.T @ K^{-1} @ y - 0.5 * log|K| - n/2 * log(2π)
"""

import warnings

import numpy as np
from scipy import linalg

from gpy._utils._computation import compute_lower_cholesky_decomposition
from gpy._utils._data import (
    normalize_input_data,
    normalize_target_data,
)
from gpy._utils._errors import ValidationError
from gpy._utils._types import Arrf64
from gpy._utils._validation import (
    validate_input_and_target_data,
    validate_numeric_array,
    validate_variable_names,
)
from gpy.Kernels._base import Kernel
from gpy.Optimization.gaussian_process.optimization import (
    optimize_hyperparameters,
)


class GaussianProcess:
    """
    Gaussian Process model for regression with optional hyperparameter
    optimization and input/output normalization.

    Attributes:
        - kernel (Kernel): The kernel function used to compute covariance.
        - x_train (Arrf64): Training input features after normalization.
        - y_train (Arrf64): Training target values after normalization.
        - alpha (Arrf64): Weights computed during fitting for predictions.
    """

    def __init__(self, kernel: Kernel, normalize_x: bool = True) -> None:
        """
        Initializes a Gaussian Process model with the specified kernel.

        Args:
            - kernel (Kernel): A kernel instance defining the covariance
                               function.
            - normalize_x (bool): Whether to normalize input features to zero
                                  mean and unit variance. Defaults to True.

        Raises:
            ValidationError: If kernel is not a valid Kernel subclass.
        """
        if not isinstance(kernel, Kernel):
            err_msg = (
                "Error: 'kernel' argument must be a valid kernel subclass."
            )
            raise ValidationError(err_msg)

        self.kernel = kernel
        self._normalize_x = normalize_x

        # simulates gaussian noise in data, optimized with kernel
        # hyperparameters helps stabilize fitting with numerically unstable data
        self._noise = 1e-3

        # training and testing data
        self.x_train = np.array([])
        self.y_train = np.array([])
        self.x_test = np.array([])
        self.y_test = np.array([])

        # GP weight parameter (alpha = K^-1 @ y)
        self.alpha = np.array([])

        # lower cholesky decomposition of the kernel matrix K
        self._lower_chol = np.array([])

        # target normalization stats (always used)
        self._y_mean = 0
        self._y_std = 1

        # input normalization stats (arrays to handle multiple features)
        # only used if normalize_x is True; otherwise set to identity transform
        if self._normalize_x:
            self._x_mean = np.array([])
            self._x_std = np.array([])

        return

    def optimize_hyperparameters(
        self, objective: str, num_restarts: int = 5
    ) -> None:
        """
        Optimizes the kernel hyperparameters using the specified objective
        function.

        Args:
            - objective (str): The objective function to minimize. Options
                               include 'lml' (log marginal likelihood).
            - num_restarts (int): Number of random restarts to avoid local
                                  minima. Defaults to 5.
        """
        optimize_hyperparameters(self, objective, num_restarts)

    def _fit_without_optimization(self) -> None:
        """
        Fits the Gaussian Process to training data without optimizing
        hyperparameters. Computes the kernel matrix, its Cholesky decomposition,
        and the alpha weights used for prediction.
        """
        # kernel matrix K(x_train, x_train)
        K = self.kernel._compute(self.x_train, self.x_train)

        self._lower_chol, self._noise = compute_lower_cholesky_decomposition(
            K, self._noise, max_attempts=10
        )

        self.alpha = linalg.cho_solve((self._lower_chol, True), self.y_train)

        return

    def fit(
        self,
        x: Arrf64,
        y: Arrf64,
        optimize: bool = False,
        objective: str = "lml",
    ) -> None:
        """
        Fits the Gaussian Process model to training data.

        Args:
            - x (Arrf64): Input features of shape (n_samples, n_features).
            - y (Arrf64): Target values of shape (n_samples,).
            - optimize (bool): Whether to optimize hyperparameters before
                               fitting. Defaults to False.
            - objective (str): Objective function for optimization if optimize
                               is True. Defaults to 'lml'.

        Raises:
            ValidationError: If input and target arrays have incompatible
                             shapes or contain invalid values.
        """
        x, y = validate_input_and_target_data(x, y)

        if self._normalize_x:
            self.x_train, self._x_mean, self._x_std = normalize_input_data(x)

        else:
            self.x_train = x
            self._x_mean = np.zeros(x.shape[1])
            self._x_std = np.ones(x.shape[1])

        self.kernel._validate_anisotropic_hyperparameter_shape(x)

        self.y_train, self._y_mean, self._y_std = normalize_target_data(y)

        if optimize:
            self.optimize_hyperparameters(objective)
            self._fit_without_optimization()

        else:
            self._fit_without_optimization()

    def predict(
        self, x: Arrf64, return_std: bool = False, return_cov: bool = False
    ) -> Arrf64 | tuple[Arrf64, ...]:
        """
        Predicts target values for new input data with optional uncertainty
        estimates.

        Args:
            - x (Arrf64): Input features of shape (n_samples, n_features).
            - return_std (bool): Whether to return standard deviation of
                                 predictions. Defaults to False.
            - return_cov (bool): Whether to return full covariance matrix.
                                 Defaults to False.

        Returns:
            Arrf64 | tuple[Arrf64, ...]: Predicted mean values, and optionally
                standard deviation and/or covariance matrix depending on flags.

        Raises:
            RuntimeError: If the model has not been fitted before prediction.
            ValidationError: If input contains invalid values.
        """
        # model must be fitted before prediction
        if self.alpha.size == 0 or self._lower_chol.size == 0:
            err_msg = (
                "Error: Model needs to be fitted before it can be used for "
                "prediction."
            )
            raise RuntimeError(err_msg)

        x = validate_numeric_array(
            x, "Gaussian Process Prediction input", allow_nonpositive=True
        )
        x = x.reshape(-1, 1) if x.ndim == 1 else x

        # normalize new input data
        # this is safe even if _normalize_x is False because self._x_mean and
        # self._x_std are initialized to default values of 0 and 1 anyways
        x_norm = (x - self._x_mean) / self._x_std

        # compute mean prediction: μ* = K(x*, X) @ α
        k_test_train = self.kernel.compute(x_norm, self.x_train)

        y_mean_norm = k_test_train @ self.alpha

        # unnormalize y_mean for returning
        y_mean = (y_mean_norm * self._y_std) + self._y_mean

        if not (return_std or return_cov):
            return y_mean

        # compute variance / covariance
        # v = L^-1 * k_test_train.T
        variance = linalg.solve_triangular(
            self._lower_chol, k_test_train.T, lower=True
        )

        if return_cov:
            k_test_test = self.kernel.compute(x_norm, x_norm)

            y_cov_norm = k_test_test - variance.T @ variance
            y_cov = y_cov_norm * (self._y_std**2)
            if return_std:
                y_std = np.sqrt(np.maximum(np.diag(y_cov), 0.0))
                return y_mean, y_std, y_cov
            return y_mean, y_cov

        k_diag = np.ones(x_norm.shape[0])

        y_var_norm = k_diag - np.sum(variance**2, axis=0)

        y_var_norm = np.maximum(y_var_norm, 0.0)
        y_std = np.sqrt(y_var_norm) * self._y_std

        return y_mean, y_std

    def to_str(self, variable_names: list[str]) -> str:
        """
        Generates a string representation of the fitted Gaussian Process as a
        mathematical expression.

        Args:
            - variable_names (list[str]): Names of input variables to use in
                                          the expression (e.g., ['x', 'y']).

        Returns:
            str: Mathematical expression representing the GP prediction
                 function.

        Warns:
            UserWarning: If the model has not been fitted.

        Raises:
            ValidationError: If variable_names length doesn't match number of
                             features.
        """
        if self.alpha.size == 0:
            warning_msg = (
                "Warning: Gaussian Process is not fitted, returning empty "
                "string."
            )
            warnings.warn(warning_msg)
            return ""

        variable_names = validate_variable_names(
            variable_names, self.x_train.shape[1]
        )

        # normalize variables to match normalized data
        # if the x normalization option was set to false, the mean and std
        # are set to 0 and 1 respectively, this will not affect the data in that
        # case
        normalized_vars = []
        for i, var in enumerate(variable_names):
            normalized_vars.append(
                f"(({var} - {self._x_mean[i]:.6e}) / {self._x_std[i]:.6e})"
            )

        terms = []
        for x_i, alpha_i in zip(self.x_train, self.alpha):
            k_str = self.kernel._to_str(normalized_vars, alpha_i, x_i)
            terms.append(k_str)

        # target data is always normalized, this unnormalizes it
        full_expression = " + ".join(terms)
        return f"({full_expression}) * {self._y_std:.6e} + {self._y_mean:.6e}"
