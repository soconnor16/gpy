"""
This module is used to wrap some of the functionality from _condition_utils and
_validation_utils to make processing data before computation easier.
"""

from gpy.core._types import Arr64, FloatSeq

from ._condition_utils import _condition_arrays
from ._validation_utils import _validate_array


def _process_arrays(
    arr1: FloatSeq,
    arr2: FloatSeq,
    name1: str,
    name2: str,
    allow_negative: bool,
) -> tuple[Arr64, Arr64]:
    """
    Function to preprocess pairs of arrays as needed in many computations.

    This function is used in the important array arithmetic computations needed
    for GPR (eg. kernel matrix calculation, mean function calculation, etc.)
    and is solely used for convenience and less repetitive code.

    Args:
        arr1 (FloatSeq): First array to be processed.
        arr2 (FloatSeq): Second array to be processed.
        name1 (str): "name" representing arr1's use for more descriptive error
                      messages during validation.
        name2 (str): "name" representing arrw's use for more descriptive error
                      messages during validation.
        allow_negative (bool): Whether the arrays should allow negative values.

    Returns:
        tuple(Arr64, Arr64): tuple of processed arrays.

    """

    arr1 = _validate_array(arr1, name=name1, allow_negative=allow_negative)
    arr2 = _validate_array(arr2, name=name2, allow_negative=allow_negative)

    return _condition_arrays(arr1, arr2)
