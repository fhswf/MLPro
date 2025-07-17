## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.math.statistics
## -- Module  : boundaries
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-06-01  0.1.0     DA       Creation 
## -- 2025-06-04  0.2.0     DA       New methods create_boundaries(), _create_boundaries()
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2025-06-04)

This module provides classes for the standardized use of value boundaries.
"""

from typing import Union
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray



# Export list for public API
__all__ = [ 'Boundaries',
            'BoundarySide',
            'BoundaryProvider' ]

            


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
    @staticmethod
    def create_boundaries( p_num_dim : int ) -> Boundaries:
        """
        Static service method to create a two-dim array for boundaries.
        """

        return np.full( (p_num_dim,2), np.nan)
    

## -------------------------------------------------------------------------------------------------
    def _create_boundaries( self, p_num_dim : int ) -> Boundaries:
        """
        Internal service method to create a two-dim array for boundaries.
        """

        return self.__class__.create_boundaries( p_num_dim = p_num_dim )


## -------------------------------------------------------------------------------------------------
    def get_boundaries( self, 
                        p_dim : int = None,
                        p_side : BoundarySide = None,
                        p_copy : bool = False ) -> Union[Boundaries, float]:
        """
        Returns the current value boundaries of internally stored data. The result can be reduced
        by the optional parameters p_side, p_dim. If both parameters are specified, the result is
        a float.

        Parameters
        ----------
        p_dim : int = None
            Optionally reduces the result to a particular dimension.
        p_side : BoundarySide = None
            Optionally reduces the result to upper or lower boundaries. See class BoundarySide for
            possible values.
        p_copy : bool = False
            If True, a copy of the boudaries is returned. Otherwise (default), a reference to the
            internal boundary array is returned.

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
