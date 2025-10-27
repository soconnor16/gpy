"""
This module generates string representations of trained Gaussian Process
Regression (GPR) models.

Mathematical Overview:
    Given a Gaussian Process trained on data sampled from an underlying
    function f, the posterior predictive mean of the model at a new input x
    can be expressed as:

        f(x) ≈ Σ αᵢ * K(x, xᵢ)

    where:
        - K(x, xᵢ) represents the kernel function between the new input x and
          each training input xᵢ.

        - α = (K + σ²I)⁻¹ y is the vector of learned coefficients obtained
          during training, where:
              K  - the kernel (covariance) matrix over training inputs
              σ²I - the Gaussian noise term for numerical stability
              y  - the observed target values
"""

import numpy as np

from .gaussian_process import GaussianProcess


class StringGenerator:
    def __init__(self, gp: GaussianProcess, variables: str | list[str]):
        self.gp = gp
        self.variables = (
            [variables] if isinstance(variables, str) else variables
        )

    def _format_distance_sq(self, x_i) -> str:
        """
        Function to format squared distance between two points.

        This function is used to differentiate how squared distance is
        represented depending on dimension, expanding the representation to
        match the number of variables passed to the generator.

        Returns:
            str: A string representation of the formatted square distance for a
                 point xᵢ.
        """
        if len(self.variables) == 1:
            return f"({self.variables[0]} - {x_i[0]:.6e})^2"
        return " + ".join(
            f"({v} - {xi:.6e})^2" for v, xi in zip(self.variables, x_i)
        )

    def _kernel_expr(self, kernel, x_i):
        """
        Function to generate the kernel expressions.

        This function is used to parse the type of the kernel that was passed to
        the generator, and generate the string representation of the kernel
        function K(x, xᵢ) for a given xᵢ.
        """
        kernel_type = type(kernel).__name__

        if kernel_type == "RBFKernel":
            sigma_sq = kernel.sigma**2
            length_sq = kernel.length_scale**2
            dist_sq = self._format_distance_sq(x_i)
            return f"{sigma_sq:.6e} * exp(-0.5 * ({dist_sq}) / {length_sq:.6e})"

        elif kernel_type == "PeriodicKernel":
            sigma_sq = kernel.sigma**2
            length_sq = kernel.length_scale**2
            period = kernel.period

            # the periodic kernel uses absolute, not square distance
            if len(self.variables) == 1:
                dist = f"abs({self.variables[0]} - {x_i[0]:.6e})"
            else:
                dist_sq = self._format_distance_sq(x_i)
                dist = f"sqrt({dist_sq})"

            pi = np.pi
            return (
                f"{sigma_sq:.6e} * exp(-2.0 * sin({pi:.6e} * ({dist}) / "
                f"{period:.6e})^2 / {length_sq:.6e})"
            )

        elif kernel_type == "ConstantKernel":
            return f"{kernel.c:.6e}"

        elif kernel_type == "CompositeKernel":
            # recursively generates the kernel strings from the substituent
            # kernels of a composite kernel, then joins the expressions with
            # the composite kernel's operation
            sub_exprs = [self._kernel_expr(k, x_i) for k in kernel.kernels]
            op = " + " if kernel.operation.name == "ADD" else " * "
            return "(" + op.join(sub_exprs) + ")"

        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

    def generate(self) -> str:
        """
        Method to generate the complete kernel string representation.

        This method iteratively generates the kernel strings for each xᵢ in the
        given GP, and joins them with a "+" to represent the summation.

        Returns:
            str: Full predictive function string representing Σ αᵢ * K(x, xᵢ)
                 for the model.
        """
        terms = []
        for x_i, alpha_i in zip(self.gp.x_train, self.gp.alpha):
            k_expr = self._kernel_expr(self.gp.kernel, x_i)
            terms.append(f"{alpha_i:.6e} * {k_expr}")

        return " + ".join(terms)
