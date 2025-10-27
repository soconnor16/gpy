from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

"""
Arbitrary Numpy array of any dimension with a data type of 64 bit float types.
This is used for almost all functions that include Matrix operations.
"""
Arr64 = NDArray[np.float64]

"""
Add-on to Python's native Sequence type hint. This program expects numpy arrays
for much of its usage, but attempts to accept, validate, and then convert valid
Python natives (eg. list of floats -> NDarray[float64]) for more flexible usage.
This "FloatSeq" type is used for type hinting to allow for this flexibility.
"""
FloatSeq = Arr64 | Sequence[float]

"""
Numeric type to encompass both Numpy and Python integer and float types.
"""
Numeric = int | np.integer | float | np.floating
