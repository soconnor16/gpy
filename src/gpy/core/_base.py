"""
This module acts as the "base" for the implementation of more specific kernels.
The Kernel abstract base class serves to outline the function of individual
kernel implementations and define the bare minimum functionality a specific
kernel must implement.
"""

from abc import ABC, abstractmethod

from ._types import Arr64, FloatSeq


class Kernel(ABC):
    @property
    @abstractmethod
    def PARAM_ORDER(self) -> tuple[str, ...]:
        """
        Property to define the order in which kernel hyperparameters are
        expected. Can be implemented as a static property for simple kernels
        or dynamic property for composite kernels.
        """

    @property
    @abstractmethod
    def BOUNDS(self) -> tuple[tuple[float, float], ...]:
        """
        Property to define the bounds applied to each hyperparameter during
        optimization. The bounds are defined in the order corresponding to
        the PARAM_ORDER property.
        """

    @abstractmethod
    def compute(self, arr1: FloatSeq, arr2: FloatSeq) -> Arr64:
        """
        Method to compute a kernel's similarity matrix between two given input
        arrays.

        Kernel functions are the core of Gaussian Process Regression (GPR).
        They are used to define the covariance (correlation) between input
        points, serving as the bedrock for the probabilistic nature of GPR.

        For the computation of a covariance matrix, two arrays may have
        different numbers of samples, but must have the same number of
        features. Given x1 with shape (n_samples1, n_features) and x2 with
        shape (n_samples2, n_features), the resultant covariance matrix
        k(x1, x2) will have shape (n_samples1, n_samples2).

        Args:
            arr1 (FloatSeq): First array of points to compute the kernel matrix
                             between. Expected as an array or array-like data
                             structure of 64 bit floats
            arr2 (FloatSeq): Second array of points to compute the kernel
                             matrix between. Expected as an array or
                             array-like data structure of 64 bit floats.

        Returns:
                (Arr64): Kernel covariance matrix calculated between arr1 and
                         arr2. Returned as a Numpy array of 64 bit floats.

        """

    @abstractmethod
    def get_params(self) -> list[float]:
        """
        Method to obtain the current hyperparameters of the kernel in the order
        given by the 'PARAM_ORDER' Class Variable.

        Returns:
            List[float]: Kernel hyperparameters as a List of Python floats.

        """

    @abstractmethod
    def set_params(self, params: FloatSeq) -> None:
        """
        Method to modify the hyperparameters of the model.

        Args:
            params (FloatSeq): Values that the hyperparameter values should be
                               set to. Expected as an array or array-like data
                               structure of 64 bit floats. Hyperparameters will
                               be set according to the order layed out in the
                               'PARAM_ORDER' Class Variable.

        """
