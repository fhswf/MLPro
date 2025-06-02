## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.math.statistics
## -- Module  : boundaries
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-06-01  1.0.0     DA       Creation 
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-05-31)

This module provides classes for statistical functionalities.
"""

from typing import Union
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray



Boundaries = NDArray[np.float64]



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BoundarySide(IntEnum):
    LOWER = 0
    UPPER = 1





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BoundaryProvider:
    """
    Standardizes the provision/computation of boundaries.
    """

## -------------------------------------------------------------------------------------------------
    def get_boundaries(self, p_side : BoundarySide = None, p_dim : int = None ) -> Union[Boundaries, float]:
        """
        Returns the current value boundaries of internally stored data. The result can be reduced
        by the optional parameters p_side, p_dim. If both parameters are specified, the result is
        a float.

        Parameters
        ----------
        p_side : BoundarySide = None
            Optionally reduces the result to upper or lower boundaries. See class BoundarySide for
            possible values.
        p_dim : int = None
            Optionally reduces the result to a particular dimension.

        Returns
        -------
        Union[Boundaries, float]
            Returns the current boundaries of the data. The return value depends on the combination 
            of the optional parameters:
            - If neither `p_side` nor `p_dim` is specified: returns the full 2×n array.
            - If only `p_side` is specified: returns a 1D array with values for all dimensions.
            - If only `p_dim` is specified: returns a 1D array with both lower and upper bounds.
            - If both `p_side` and `p_dim` are specified: returns a single float value.
        """
        
        raise NotImplementedError
