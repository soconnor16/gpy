"""
This module acts as the mathematical framework that allows for the usage of
Gaussian Processes for regression.

Mathematical Overview:
    1. GP Definition:
        A Gaussian Process (GP) is defined as a set of random variables (X)
        indexed by a set T such that any finite collection of these variables
        has a joint multivariate normal distribution. That is, for any
        F = {t₁,t₂,...,tₙ} ⊂ T, the vector X_F = {Xₜ₁,Xₜ₂ ,...,Xₜₙ}
        satisfies: X ~ N(µ_F ,Σ_F)
    2. Predictive Function:
        GPR returns well defined predictive functions f(X). They often take the
        form f(X) ~ N(µ(X) ,Σ(X)) or f(X) = Σ ⍺ᵢ * K(x, xᵢ) where
        ⍺ = {⍺₁,⍺₂,...,⍺ₙ} = (K + σ²I)⁻¹ @ y where σ²I accounts for gaussian
        noise, K represents our kernel function, and X, y represent our input
        and target data respectively.

GP Properties:
    1. Defined by Mean Vector and Covariance matrix:
        A 1D Gaussian function is completely described by its mean µ and its
        variance σ, both of which are scalar values. This property is
        generalized for the multivariate Gaussian Distributions used in GPs.
        GPs are instead described by their mean vector and covariance matrix.
        In the context of GPR, the mean vector is instead interpreted as a mean
        function which when created, can be thought of as the most likely
        output values of the given input points. The covariance matrix, or in
        the context of GPR, kernel matrix represents the similarity between
        any two input points and is defined by the chosen kernel.
    2. Time complexity:
        GPR requires the inversion of the Kernel matrix which largely
        contributes to its O(n³) time complexity. This makes GPR largely
        impractical for large datasets (~10,000 training points depending on the
        hardware), after this point, it is likely better to utilize different
        regression methods.

"""

import numpy as np
from scipy import linalg, optimize

from gpy._utils._condition_utils import _condition_array
from gpy._utils._validation_utils import _validate_array

from ._base import Kernel
from ._types import Arr64


class GaussianProcess:
    def __init__(self, kernel: Kernel) -> None:
        if not isinstance(kernel, Kernel):
            raise TypeError(
                "Error: 'kernel' must be a Kernel instance. "
                f"Got: {type(kernel).__name__}.",
            )

        self.kernel = kernel

        # value to simulate gaussian noise that can be expected in data
        self._noise = 1e-3

        # test and training data, initialized to empty arrays
        self.x_train = np.array([])
        self.y_train = np.array([])
        self.x_test = np.array([])
        self.y_test = np.array([])

        # alpha values which can be thought of as training value weights,
        # initialized to empty arrays
        self.alpha = np.array([])

        # L represents the lower triangular cholesky decomposition of K
        self.L = np.array([])

        # y_train mean and std for normalization, initialized to default values
        self._y_mean = 0
        self._y_std = 1

    def _fit_without_optimization(self) -> None:
        K = self.kernel.compute(self.x_train, self.x_train)
        # ensure K is symmetric
        K = (K + K.T) / 2
        noise = self._noise

        # cholesky decomposition can be unstable, this attemps automatic
        # correction
        max_attempts = 10
        for attempt in range(max_attempts):
            # first strategy: add minimal noise to the diagonal to stabilize
            try:
                K_noise = K + noise * np.eye(K.shape[0])
                self.L = linalg.cholesky(K_noise, lower=True)
                break

            except linalg.LinAlgError:
                # second strategy: try to add scaled noise based on matrix diag
                if attempt < max_attempts - 1:
                    k_scale = float(np.mean(np.diag(K)))
                    noise = max(10 * noise, k_scale * noise)

                else:
                    # final strategy: scale based on eigenvalues
                    try:
                        K_noise = K + noise * np.eye(K.shape[0])
                        eigvals = np.linalg.eigvalsh(K_noise)
                        min_eig = np.min(eigvals)

                        if min_eig < 0.0:
                            jitter = abs(min_eig) + 1e-6
                            K_noise = K + (noise + jitter) * np.eye(K.shape[0])
                            self.L = linalg.cholesky(K_noise, lower=True)

                        else:
                            raise ValueError(
                                "Error: Cholesky decomposition failed despite "
                                "positive eigenvalues."
                            )
                    except linalg.LinAlgError as exc:
                        raise ValueError(
                            "Error during fitting: Numerical instability "
                            f"during matrix decomposition: {exc!s}"
                        )

        self.alpha = linalg.cho_solve((self.L, True), self.y_train)

    def _compute_log_marginal_likelihood(self) -> float:
        num_samples = self.x_train.shape[0]

        data_fit_term = -0.5 * self.y_train @ self.alpha
        complexity_penalty = -np.sum(np.log(np.diag(self.L)))
        constant_term = -0.5 * num_samples * np.log(2 * np.pi)

        return float(data_fit_term + complexity_penalty + constant_term)

    def optimize_hyperparameters(self) -> None:
        """
        Method to optimize the hyperparameters of the model by maximizing
        its log-likelihood. Log-likelihood was chosen as the objective function
        because of its natural ties to probabalistic regression models.
        """
        initial_kernel_params = self.kernel.get_params()
        initial_noise = self._noise
        initial_params = np.array([*initial_kernel_params, initial_noise])

        def objective(params: Arr64) -> float:
            kernel_params = params[:-1]
            noise = params[-1]

            self.kernel.set_params(kernel_params)
            self._noise = float(noise)

            self._fit_without_optimization()

            return -self._compute_log_marginal_likelihood()

        # second tuple of bounds are for noise value
        bounds = [*list(self.kernel.BOUNDS), (1e-5, 1e1)]

        result = optimize.minimize(
            objective,
            initial_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "disp": False},
            tol=1e-6,
        )

        final_params = result.x
        self.kernel.set_params(final_params[:-1])
        self._noise = final_params[-1]
        self._fit_without_optimization()

    def fit(
        self,
        x_train: Arr64,
        y_train: Arr64,
        optimize_params: bool = False,
    ) -> None:
        """
        Method to fit the model with its training data.

        This method performs the fitting step of Gaussian Process Regression
        (GPR). The input training data are standarized (mean of 0 and standard
        deviation of 1) to improve numerical stability and make hyperparameter
        optimization more consistent.

        GPR models the training data as samples from a joint Gaussian
        distribution over possible function values. Instead of fixed parameters,
        GPR defines a mean function m(x) (representing the expected output value
        for each input) and a kernel function k(x,x') (representing the
        covariance between function values at different inputs).

        Fitting a Gaussian Process involves computing the posterior distribution
        of the function given the training data. This posterior is completely
        described by the posterior mean function (the most probable prediction
        value for each input) and a posterior covariance matrix (the model's
        uncertainty about those predictions), both derived from the prior mean
        and kernel functions and the observed training data.

        Args:
            x_train (Arr64): Training input values
            y_train (Arr64): Training output values
            optimize_params (bool): Whether to optimize the hyperparameters of
                                    the model after fitting.
        """
        self.x_train = _validate_array(x_train, "GP X train", True)
        self.x_train = _condition_array(self.x_train)

        y_train = _validate_array(y_train, "GP y train", True)

        self._y_mean = np.mean(y_train)
        self._y_std = np.std(y_train)
        if self._y_std == 0:
            self._y_std = 1e-3

        y_train_norm = (y_train - self._y_mean) / self._y_std
        self.y_train = y_train_norm

        if (y_train.ndim != 1) or (y_train.shape[0] != self.x_train.shape[0]):
            raise ValueError(
                "Error: 'y_train' should be 1D and have the same number of "
                "samples as 'x_train'",
            )

        if optimize_params:
            self.optimize_hyperparameters()
        else:
            self._fit_without_optimization()

    def predict(
        self,
        x_test: Arr64,
        return_variance: bool = True,
    ) -> tuple[Arr64, ...]:
        """
        Method to make predictions using the fitted model.

        This method uses the posterior distribution created during model fitting
        to predict output values for new input data points.

        In GPR, once a posterior mean function m*(x_*) and covariance function
        k*(x_*, x'_*) are determined from the training data, predictions at new
        input points are obtained as conditional distribiutions derived from
        the joint Gaussian prior.

        Specifically, for test inputs X*, the predictive mean and covariance are
        computed as:
            μ* = K(X_*, X)[K(X,X) + (𝛔^2)I]^-1 @ y
            Σ* = K(X_*, X_*) - K(X*, X)[K(X,X) + (𝛔^2)I]^-1 @ K(X, X*)

        Where:
            - K(X*, X) is the kernel matrix between test and training points
            - K(X, X) is the kernel matrix among training points
            - 𝛔^2 is the gaussian noise variance
            - y are hte observed training outputs

        The result is a predictive mean (most likely output values of the new
        input data) and a predictive variance (uncertainty in the predictions).

        Args:
            x_test (Arr64): The testing input values whose predicted outputs are
                            to be given.
            return_variance (bool): Whether to return the predictive variance
                                    values.

        Returns:
            tuple(Arr64, ...): Tuple of predictive means, and optionally,
                               predictive variances.
        """
        if np.any(
            [
                arr.size == 0
                for arr in [self.x_train, self.y_train, self.L, self.alpha]
            ],
        ):
            raise ValueError("Error: Model must be fitted before prediction.")

        self.x_test = _validate_array(x_test, "GP x_test", True)
        self.x_test = _condition_array(self.x_test)

        # computes cross covariance: K(x_test, x_train)
        K_star = self.kernel.compute(self.x_test, self.x_train)

        # computes posterior mean
        mean = K_star @ self.alpha
        # unnormalize mean
        mean = mean * self._y_std + self._y_mean

        if not return_variance:
            return (mean,)

        # computes test covariance: K(x_test, x_test)
        K_test = self.kernel.compute(self.x_test, self.x_test)
        # temp term representing the solution to (L)(L^T)(temp) = K_star^T
        temp = linalg.cho_solve((self.L, True), K_star.T)
        variance = np.diag(K_test) - np.sum(K_star * temp.T, axis=1)
        # unnormalize variance
        variance = variance * (self._y_std**2)

        return mean, variance
