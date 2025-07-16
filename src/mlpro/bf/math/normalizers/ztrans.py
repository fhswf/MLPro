## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.math.normalizers
## -- Module  : ztrans.py
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
## -- 2024-05-23  1.2.0     DA       Refactoring (not yet finished)
## -- 2024-05-24  1.2.1     LSB      Bug fix for Parameter update using only p_data_del in Z-transform
## -- 2024-05-27  1.2.2     LSB      Scientific Reference added
## -- 2024-12-09  1.3.0     DA       Method NormalizerZTrans.update_parameters(): review/optimization
## -- 2025-07-05  2.0.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.0.0 (2025-07-05)

This module provides a class for Z transformation.
"""


from typing import Union

import numpy as np

from mlpro.bf.various import ScientificObject
from mlpro.bf.math import Set, Element
from mlpro.bf.math.normalizers import Normalizer



# Export list for public API
__all__ = [ 'NormalizerZTrans' ]





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerZTrans (Normalizer):
    """
    Class for Normalization based on Z transformation.
    """

    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_URL        = 'http://datagenetics.com/blog/november22017/index.html'
    C_SCIREF_ACCESSED   = '2024-05-27'

    C_EPSILON           = 1e-8

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_input_set : Set = None, 
                  p_output_set : Set = None,
                  p_output_elem_cls : type = Element,
                  p_autocreate_elements : bool = True,
                  **p_kwargs ):
        
        Normalizer.__init__( self,
                             p_input_set = p_input_set,
                             p_output_set = p_output_set,
                             p_output_elem_cls = p_output_elem_cls,
                             p_autocreate_elements = p_autocreate_elements,
                             **p_kwargs )
        
        self._n         = 0
        self._mean      = None
        self._s         = None
        self._std       = None


## -------------------------------------------------------------------------------------------------
    def _update_parameters(self,
                        p_dataset: np.ndarray = None,
                        p_data_new: Union[Element, np.ndarray] = None,
                        p_data_del: Union[Element, np.ndarray] = None) -> bool:
        """
        Method to update the normalization parameters for Z transformer.

        Parameters
        ----------
        p_dataset : np.ndarray, optional
            Full dataset to reset parameters from scratch.
        p_data_new : Union[Element, np.ndarray], optional
            New data element to update parameters.
        p_data_del : Union[Element, np.ndarray], optional
            Obsolete data element to remove from parameter computation.

        Returns
        -------
        bool
            True if parameters were updated successfully.
        """

        # 0 Backup current parameters
        if self._param_new is not None:
            if self._param_old is None:
                self._param_old = self._param_new.copy()
            else:
                np.copyto(self._param_old, self._param_new)


        # 1 Update on dataset (full reset)
        if p_dataset is not None:
            self._n = len(p_dataset)

            if self._mean is None or self._mean.shape != p_dataset.shape[1:]:
                self._mean      = np.zeros(p_dataset.shape[1:], dtype=np.float64)
                self._s         = np.zeros(p_dataset.shape[1:], dtype=np.float64)
                self._std       = np.ones(p_dataset.shape[1:], dtype=np.float64)
                self._mean_old  = np.zeros_like(self._mean)
                
            np.copyto(self._mean, np.mean(p_dataset, axis=0, dtype=np.float64))
            np.copyto(self._std, np.std(p_dataset, axis=0, dtype=np.float64))
            np.copyto(self._s, np.square(self._std) * self._n)

            if self._param_new is None:
                self._param_new = np.zeros([2, self._std.shape[-1]])

        else:
            # 2 Update on new data
            if p_data_new is not None:
                try:
                    data_new = np.array(p_data_new.get_values())
                except:
                    data_new = p_data_new

                if self._n == 0:
                    # 2.1 First call
                    self._n        = 1
                    self._mean     = np.array(data_new, dtype=np.float64)
                    self._s        = np.zeros_like(data_new, dtype=np.float64)
                    self._std      = np.ones_like(data_new, dtype=np.float64)
                    self._mean_old = np.zeros_like(data_new, dtype=np.float64)
                    
                    if self._param_new is None:
                        self._param_new = np.zeros([2, data_new.shape[-1]])

                    np.copyto(self._param_new[0], 1.0)
                    np.copyto(self._param_new[1], -self._mean)

                    self._set_parameters(p_param=self._param_new)
                    return True

                # 2.2 Incremental update
                self._n += 1
                np.copyto(self._mean_old, self._mean)
                np.copyto(self._mean, self._mean_old + (data_new - self._mean_old) / self._n)
                np.add(self._s, (data_new - self._mean) * (data_new - self._mean_old), out=self._s)
                np.copyto(self._std, np.sqrt(self._s / self._n) )


            # 3 Update on obsolete data
            if p_data_del is not None:
                try:
                    data_del = np.array(p_data_del.get_values())
                except:
                    data_del = p_data_del

                if self._n > 1:
                    self._n -= 1
                    np.copyto(self._mean_old, self._mean)
                    np.copyto(self._mean, self._mean_old - (data_del - self._mean_old) / self._n )
                    np.subtract(self._s, (data_del - self._mean) * (data_del - self._mean_old), out=self._s)
                    np.copyto(self._std, np.sqrt(self._s / self._n))

                else:
                    self._n = 0
                    np.copyto(self._mean, 0.0)
                    np.copyto(self._s, 0.0)
                    np.copyto(self._std, 1.0)

                    np.copyto(self._param_new[0], 1.0)
                    np.copyto(self._param_new[1], -self._mean)

                    self._set_parameters(p_param=self._param_new)
                    return True


        # 4 Update of parameters with epsilon safeguard
        safe_std = np.maximum(self._std, self.C_EPSILON)
        np.copyto(self._param_new[0], np.divide(1, safe_std, out=np.zeros_like(safe_std), where=safe_std != 0))
        np.copyto(self._param_new[1], -np.divide(self._mean, safe_std, out=np.zeros_like(safe_std), where=safe_std != 0))

        self._set_parameters(p_param=self._param_new)
        return True
