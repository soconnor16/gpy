"""
This module provides an active learning strategy to automate the efficient
creation of accurate GPR models.

Active Learning Overview:
    In the context of machine learning, active learning represents the idea of
    creating an acquisition function which can automatically choose training
    points from an unlabeled dataset. Its purpose is to both automate the
    process of model creation, as well as create the model more efficiently
    (increased accuracy with a lower number of total training points).

    The active learning process often works like so:
        1. Initial model:
            An initial model is created and trained on a small set of labeled
            (training) points. These points are manually chosen, or chosen via
            set criteria (ie. a random set of points, points that are evenly
            spaced to span the whole range of points, etc.)

        2. Model Quantification:
            The model is used for prediction which can be used to quantify the
            model's accuracy and, in the case of GPR, uncertainty. If the
            accuracy of the model is sufficient, training stops here.

        3. Acquisition function:
            An acquisition function is used to identify which of the remaining
            unlabeled data points should be chosen and added to the training
            pool for the next round of training.

        4. Point Selection:
            A new point (or multiple points if done in batches) is added to the
            training pool according to the acquisition function.

        5. Training:
            The GPR model is retrained using the new training pool.

        6. Repeat:
            This process is iterative, steps 2-5 are repeated until a desired
            model accuracy is reached.

Implementation In This Module:
    Acquisition function:
        In this active learning implentation, points are chosen using variance
        values. The variance value of a training point is given by the value of
        the diagonal of the model's covariance matrix that corresponds with the
        point. This variance value can also be interpreted as the uncertainty or
        "confidence" of the model at that point. High variance values indicate
        points where the model has low confidence, points where it believes the
        true value may actually lie in a larger range around the predicted
        value. The essential idea of centering the acquisition function
        around variance values is the hope that they will provide the model
        information where it needs it the most as opposed to possibly
        overtraining certain regions while ignoring others.

    Stopping Criteria:
        RMSE threshold:
            One of the stopping criteria for this active learning implementation
            is based on an rmse threshold. There is a sane default value that
            should work for most uses, but it can be manually configured when
            creating the learner should the need arise. The rmse of the model is
            calculated across the whole dataset so that the entire function is
            considered.

        Number of Points:
            Another stopping criterion implemented is a maximum number of points
            for the model to use in its training pool. There is also a sane
            default implemented for this (80% of the total number of points).
            This should also work for most datasets, but can be manually
            configured.

    Model Optimization:
        During the iterative learning cycle, the model's hyperparameters are
        optimized after each retraining. This is done by maximizing
        marginal-log-likelihood until the rmse stopping threshold is reached,
        after which the cost function is changed and the hyperparameters are
        optimized to minimize rmse. This combination is introduced as an attempt
        to prevent overtraining while also benefiting from the more aggresive
        rmse based optimization.
"""

from dataclasses import dataclass, field
from math import floor

import numpy as np
from numpy._typing import NDArray
from scipy import linalg, optimize

from gpy._utils._condition_utils import _condition_array
from gpy._utils._validation_utils import (
    _validate_array,
    _validate_numeric_value,
)

from ._base import Kernel
from ._types import Arr64, Numeric
from .gaussian_process import GaussianProcess


@dataclass
class TrainingHistory:
    """
    Dataclass used for storing the training history of the learner. For each
    iteration, it stores the index chosen, the rmse value of the model when that
    point was chosen, and the variance of the point chosen.
    """

    selected_indices: list[int] = field(default_factory=list)
    rmse_values: list[float] = field(default_factory=list)
    variances: list[float] = field(default_factory=list)


class ActiveLearner:
    def __init__(
        self,
        kernel: Kernel,
        x_full: Arr64,
        y_full: Arr64,
        max_points: int | None = None,
        rmse_threshold: Numeric = 0.5,
        optimize_interval: int | None = 1,
    ) -> None:
        self.kernel = kernel
        self.x_full = _condition_array(
            _validate_array(x_full, "X full", allow_negative=True)
        )
        self.y_full = _validate_array(
            y_full, "y full", allow_negative=True
        ).ravel()  # make sure y_full is 1D

        # y_full must have the same number of samples as x_full
        if self.y_full.shape[0] != self.x_full.shape[0]:
            raise ValueError(
                "Error: 'y_full' should have the same number of features as "
                "'x_full'\n\n"
                f"y_full shape (flattened): {self.y_full.shape}\n"
                f"x_full shape: {self.x_full.shape}"
            )

        self.gp = GaussianProcess(self.kernel)

        self.rmse_threshold = float(
            _validate_numeric_value(
                rmse_threshold, "rmse threshold", allow_negative=False
            )
        )

        if max_points:
            self.max_points = int(
                _validate_numeric_value(
                    max_points, "max points", allow_negative=False
                )
            )
        else:
            self.max_points = floor(len(self.y_full) * 0.8)

        # this covers the possible division by zero error in the
        # iteration % self.optimize interval check too
        if not optimize_interval:
            self.optimize_interval = None
        else:
            self.optimize_interval = int(
                _validate_numeric_value(
                    optimize_interval, "optimize_interval", allow_negative=False
                )
            )

        # initialize training sets and pool of points that haven't been picked
        self.x_train = np.array([])
        self.y_train = np.array([])
        self.remaining_indices = np.array([])

        self._initialize_training_data()

    def _initialize_training_data(self) -> None:
        """
        Function to initialize the training pool based on the datasets passed.

        The training points are initialized using three points (the first,
        middle, and last points/samples of the dataset). A pool of future
        candidates is then created to keep track of what points are available
        to be chosen by future iterations. During initialization, this pool
        consists of the entire dataset, bar the three initially chosen points.
        """
        # validate that at least 3 samples are in x_full for initial indices
        num_samples = self.x_full.shape[0]

        if num_samples < 3:
            raise ValueError(
                "Error: 'x_full' must have at least three samples for "
                f"initialization. Found {num_samples}"
            )

        # first, middle, and last points of the dataset
        initial_indices = [0, num_samples // 2, num_samples - 1]

        self.x_train = self.x_full[initial_indices]
        self.y_train = self.y_full[initial_indices]

        # remove initial indices from the training pool
        self.remaining_indices = np.setdiff1d(
            np.arange(num_samples), initial_indices
        )

    def _select_next_point(self) -> tuple[int, float]:
        """
        Selection function used to determine which point to add to the training
        pool during a training iteration.

        This selection function is relatively simple, it simply uses the current
        model to predict on the pool of remaining indices. From this, the point
        which has the largest variance value is chosen and added to the training
        pool for the next iteration.
        """
        # predict on training pool and collect each points variance
        _, variances = self.gp.predict(
            self.x_full[self.remaining_indices], return_variance=True
        )

        max_var_idx = int(np.argmax(variances))
        selected_idx = int(self.remaining_indices[max_var_idx])
        selected_variance = float(variances[max_var_idx])

        return selected_idx, selected_variance

    def compute_rmse(self) -> float:
        """
        Simple method to compute the current rmse of the model.
        """
        # predictions across the entire input dataset
        (y_pred,) = self.gp.predict(self.x_full, return_variance=False)

        return float(np.sqrt(np.mean((y_pred - self.y_full) ** 2)))

    def _optimize_params(self) -> float:
        """
        Function which optimizes the hyperparameters of the model to minimize
        rmse during training.

        Most of the training process uses the GaussianProcess class's log
        likelihood based optimization for hyperparameter optimization. This is
        mostly done to prevent overfitting the model, as rmse based
        hyperparameter optimization can lead to aggressive overfitting,
        especially for probabalistic models. This function is called at the end
        of the training process (if the rmse threshold is reached, if the
        maximum number of points is reached, or if the model runs out of points
        to choose) as a final optimization method to fine tune the
        hyperparameters.
        """
        initial_params = np.array(
            self.gp.kernel.get_params() + [self.gp._noise], dtype=np.float64
        )

        bounds = [*list(self.kernel.BOUNDS), (1e-5, 1e1)]

        def rmse_objective(params: NDArray[np.float64]) -> float:
            try:
                self.gp.kernel.set_params(params[:-1])
                self.gp._noise = float(params[-1])
                self.gp.fit(self.x_train, self.y_train, optimize_params=False)

                return self.compute_rmse()

            except (ValueError, linalg.LinAlgError):
                return float("inf")

        try:
            # initial optimization attempt
            result = optimize.minimize(
                rmse_objective,
                initial_params,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000, "disp": False},
                tol=1e-6,
            )

            best_rmse = result.fun
            best_params = result.x

            # secondary optimization for special cases
            if result.fun > self.rmse_threshold:
                # perturbed starting parameters may yield better results
                for _ in range(5):
                    pert_params = initial_params * (
                        1 + 0.3 * np.random.default_rng().standard_normal()
                    )

                    # parameters should always be positive
                    pert_params = np.abs(pert_params)

                    new_result = optimize.minimize(
                        rmse_objective,
                        pert_params,
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={"maxiter": 1000, "disp": False},
                        tol=1e-8,
                    )

                    if new_result.fun < best_rmse:
                        best_rmse = new_result.fun
                        best_params = new_result.x

            self.gp.kernel.set_params(best_params[:-1])
            self.gp._noise = float(best_params[-1])
            self.gp.fit(self.x_train, self.y_train, optimize_params=False)

            return self.compute_rmse()

        except linalg.LinAlgError:
            print(
                "Warning: RMSE optimization failed to to numerical instability"
            )

        return self.compute_rmse()

    def _update_training_data(
        self,
        selected_index: int,
        variance: float,
        history: TrainingHistory,
    ) -> None:
        """
        Function to update the TrainingHistory instance of the model, as well as
        update the training pool to remove the index most recently chosen before
        the next training iteration.
        """
        self.x_train = np.vstack([self.x_train, self.x_full[selected_index]])
        self.y_train = np.append(self.y_train, self.y_full[selected_index])

        self.remaining_indices = self.remaining_indices[
            self.remaining_indices != selected_index
        ]

        history.selected_indices.append(selected_index)
        history.variances.append(variance)

    def learn(self) -> TrainingHistory:
        """
        Method to execute the ActiveLearning workflow in its entirety. A more
        detailed explanation of this workflow can be found in the top docstring
        in this module.
        """
        history = TrainingHistory()
        history.selected_indices.extend(
            [0, self.x_full.shape[0] // 2, self.x_full.shape[0] - 1]
        )

        for iteration in range(self.max_points):
            should_optimize = (
                self.optimize_interval is not None
                and iteration % self.optimize_interval == 0
            )
            # step 1: fit model to training data, optimize w log likelihood
            self.gp.fit(
                self.x_train, self.y_train, optimize_params=should_optimize
            )

            # step 2: compute rmse and check if the threshold has been reached
            current_rmse = self.compute_rmse()
            history.rmse_values.append(current_rmse)

            if current_rmse <= self.rmse_threshold:
                # if it has reached the threshold, perform final optimization
                # minimizing rmse
                final_rmse = self._optimize_params()
                history.rmse_values[-1] = final_rmse
                print(
                    "\033[4mRMSE threshold reached\033[0m:",
                    f"\nFinal RMSE: {final_rmse:.4f}",
                    f"\nPoints used: {len(history.selected_indices)}",
                )

                break
            if len(self.remaining_indices) == 0:
                final_rmse = self._optimize_params()
                history.rmse_values[-1] = final_rmse

                print(
                    "\033[4mAll points used\033[0m:",
                    f"\nFinal RMSE: {final_rmse:.4f}",
                    f"\nPoints used: {len(history.selected_indices)}",
                )

                break

            if len(history.selected_indices) == self.max_points:
                final_rmse = self._optimize_params()
                history.rmse_values[-1] = final_rmse
                print(
                    "\033[4mMax iterations reached\033[0m:",
                    f"\nFinal RMSE: {final_rmse:.4f}",
                    f"\nPoints used: {len(history.selected_indices)}",
                )

                break

            try:
                selected_idx, selected_variance = self._select_next_point()
                self._update_training_data(
                    selected_idx, selected_variance, history
                )

            except ValueError as exc:
                # usually due to running out of points
                print(f"Warning: Learning stopped early: {str(exc)}")

                final_rmse = self._optimize_params()
                history.rmse_values[-1] = final_rmse

                break

        return history
