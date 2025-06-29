## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.math.normalizers
## -- Module  : minmax.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-09-16  0.0.0     LSB      Creation
## -- 2022-09-20  1.0.0     LSB      Release of first version
## -- 2022-09-23  1.0.1     LSB      Refactoring
## -- 2022-09-26  1.0.2     LSB      Refatoring and reduced custom normalize and denormalize methods
## -- 2022-10-01  1.0.3     LSB      Refactoring and redefining the update parameter method
## -- 2022-10-16  1.0.4     LSB      Updating z-transform parameters based on a new data/element(np.ndarray)
## -- 2022-10-18  1.0.5     LSB      Refactoring following the review
## -- 2022-11-03  1.0.6     LSB      Refactoring new update methods
## -- 2022-11-03  1.0.7     LSB      Refactoring
## -- 2022-12-09  1.0.8     LSB      Handling zero division error
## -- 2022-12-09  1.0.9     LSB      Returning same object after normalization and denormalization
## -- 2022-12-29  1.0.10    LSB      Bug Fix
## -- 2022-12-30  1.0.11    LSB      Bug Fix ZTransform
## -- 2023-01-07  1.0.12    LSB      Bug Fix
## -- 2023-01-12  1.0.13    LSB      Bug Fix
## -- 2023-02-13  1.0.14    LSB      BugFix: Changed the direct reference to p_param to a copy object
## -- 2024-04-30  1.1.0     DA       Refactoring/separation
## -- 2024-07-12  1.1.1     LSB      Renormalization error
## -- 2025-06-16  2.0.0     DA       Class NormalizerMinMax:
## --                                - New parameter p_dst_boundaries
## --                                - Refactoring of method update_parameters()
## -- 2025-06-28  2.1.0     DA       Class NormalizerMinMax: new numerical stabilizer C_EPSILON
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.1.0 (2025-06-28)

This module provides a class for MinMax normalization.
"""


from typing import Union

import numpy as np

from mlpro.bf.exceptions import ParamError
from mlpro.bf.math import Set
from mlpro.bf.math.normalizers import Normalizer



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerMinMax (Normalizer):
    """
    Class to normalize elements based on MinMax normalization.

    Parameters
    ----------
    p_dst_boundaries : list = [-1,1]
        Explicit list of (low, high) destination boundaries. Default is [-1, 1].
    """

    C_EPSILON   = 1e-12

# -------------------------------------------------------------------------------------------------
    def __init__( self, p_dst_boundaries : list = [-1,1]):

        super().__init__()
        self._dst_boundaries = p_dst_boundaries
        self._dst_diff       = p_dst_boundaries[1] - p_dst_boundaries[0]


# -------------------------------------------------------------------------------------------------
    def update_parameters(self, p_set: Set = None, p_boundaries: Union[list, np.ndarray] = None):
        """
        Update the normalization parameters using MinMax strategy.

        Parameters
        ----------
        p_set : Set, optional
            A set object providing dimensional boundaries per feature.

        p_boundaries : list or np.ndarray, optional
            Explicit array of (low, high) boundaries for each dimension.

        Raises
        ------
        ParamError
            Raised if neither p_set nor p_boundaries is provided.
        """

        # 1 Determine normalization boundaries
        if p_set is not None:
            dim_ids = p_set.get_dim_ids()
            boundaries = [p_set.get_dim(i).get_boundaries() for i in dim_ids]
        elif p_boundaries is not None:
            boundaries = np.asarray(p_boundaries).reshape(-1, 2)
        else:
            raise ParamError("Either p_set or p_boundaries must be provided.")

        n_dims = len(boundaries)


        # 2 Initialize or update _param_old
        if self._param_new is not None:
            if self._param_old is None or self._param_old.shape != self._param_new.shape:
                self._param_old = self._param_new.copy()
            else:
                self._param_old[...] = self._param_new


        # 3 Allocate or reuse _param_new
        if self._param_new is None or self._param_new.shape != (2, n_dims):
            self._param_new = np.zeros((2, n_dims), dtype=np.float64)
        else:
            self._param_new[:] = 0.0  # optional in-place reset


        # 4 Compute new parameters into _param_new
        for i, (low, high) in enumerate(boundaries):
            if high == low:
                high += 1
                low  -= 1

            diff = high - low
            if diff < self.C_EPSILON:
                diff = self.C_EPSILON

            p0 = self._dst_diff / diff
            p1 = self._dst_boundaries[0] - low * p0
            self._param_new[:, i] = p0, p1


        # 5 Activate _param_new by reference
        self._param = self._param_new
