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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.1 (2024-07-12)

This module provides a class for MinMax normalization.
"""


from mlpro.bf.math.normalizers.basics import *
import numpy as np
from typing import Union



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerMinMax(Normalizer):
    """
    Class to normalize elements based on MinMax normalization.
    """

    ## -------------------------------------------------------------------------------------------------
    def update_parameters(self, p_set: Set = None, p_boundaries: Union[list, np.ndarray] = None):
        """
        Method to update the normalization parameters of MinMax normalizer.

        Parameters
        ----------
        p_set:Set
            Set related to the elements to be normalized
        p_boundaries:ndarray
            array consisting of boundaries related to the dimension of the array

        """
        if self._param_new is not None: self._param_old = self._param_new.copy()

        try:
            if self._param_new is None: self._param_new = np.zeros([2, len(p_set.get_dim_ids())], dtype=np.float64)
            boundaries = [p_set.get_dim(i).get_boundaries() for i in p_set.get_dim_ids()]

        except:
            try:
                if self._param_new is None: self._param_new = np.zeros([(len(p_boundaries)), (len(p_boundaries))])
                boundaries = p_boundaries.reshape(-1, 2)

            except:
                raise ParamError("Wrong parameters provided for update. Please provide a set as p_set or boundaries as "
                                 "p_boundaries")

        for i, boundary in enumerate(boundaries):
            if (boundary[1] - boundary[0]) == 0:
                self._param_new[0][i] = 0
            else:
                self._param_new[0][i] = (2 / (boundary[1] - boundary[0]))
            if (boundary[1] - boundary[0]) == 0:
                self._param_new[0][i] = 0
            else:
                self._param_new[1][i] = (2 * boundary[0] / (boundary[1] - boundary[0]) + 1)

        if self._param is not None:
            self._param_old = self._param.copy()
        else:
            self._param_old = self._param_new.copy()
        self._param = self._param_new.copy()

