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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.2 (2024-05-27)

This module provides a class for Z transformation.
"""


from mlpro.bf.math.normalizers.basics import *
import numpy as np
from typing import Union
from mlpro.bf.various import ScientificObject



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerZTrans (Normalizer, ScientificObject):
    """
    Class for Normalization based on Z transformation.
    """
    C_SCIREF_TYPE = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_URL = 'http://datagenetics.com/blog/november22017/index.html'
    C_SCIREF_ACCESSED = '2024-05-27'

## -------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self._n = 0


## -------------------------------------------------------------------------------------------------
    def update_parameters(self,
                          p_dataset: np.ndarray = None,
                          p_data_new: Union[Element, np.ndarray] = None,
                          p_data_del: Union[Element, np.ndarray] = None):
        """
        Method to update the normalization parameters for Z transformer

        Parameters
        ----------
        p_dataset:numpy array
            Dataset related to the elements to be normalized. Using this parameter will reset the normalization
            parameters based on the dataset provided.
        p_data_new:Element or numpy array
            New element to update the normalization parameters. Using this parameter will set/update the
            normalization parameters based on the data provided.
        p_data_del:Element or Numpy array
            Old element that is replaced with the new element.

        """

        # 0 Backup current parameters
        if self._param_new is not None: 
            self._param_old = self._param_new.copy()


        # 1 Update on dataset
        if p_dataset is not None:
            self._std   = np.std(p_dataset, axis=0, dtype=np.float64)
            self._mean  = np.mean(p_dataset, axis=0, dtype=np.float64)
            self._n     = len(p_dataset)

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
                    self._n = 1
                    self._mean = data_new.copy()
                    self._std = np.zeros(shape=data_new.shape)
                else:
                    old_mean   = self._mean.copy()
                    self._mean = (old_mean * self._n + data_new) / (self._n + 1)

                    self._std = np.sqrt((np.square(self._std) * self._n
                                        + (data_new - self._mean) * (data_new - old_mean)) / (self._n+1))
                    self._n += 1
                    
                if self._param_new is None: 
                    self._param_new = np.zeros([2, data_new.shape[-1]])


            # 3 Update on obsolete data
            if ( p_data_del is not None ) and ( self._n > 0 ):
                try:
                    data_del = np.array(p_data_del.get_values())
                except:
                    data_del = p_data_del
            
                old_mean = self._mean.copy()
                self._mean = (old_mean * self._n - data_del) / (self._n-1)

                self._std = np.sqrt((np.square(self._std)*self._n - (data_del - old_mean)*(data_del - self._mean)) / (self._n-1))
                
                self._n -= 1


        # 4 Update of parameters
        self._param_new[0] = np.divide(1, self._std, out = np.zeros_like(self._std), where = self._std!=0)
        self._param_new[1] = np.divide(self._mean, self._std, out = np.zeros_like(self._std), where = self._std!=0)

        if self._param_old is None:
            self._param_old = self._param_new

        self._set_parameters( p_param = self._param_new )


## -------------------------------------------------------------------------------------------------
    def denormalize(self, p_data: Union[Element, np.ndarray]):
        """
        Method to denormalize a data (Element/ndarray) element based on MinMax or Z-transformation

        Parameters
        ----------
        p_data:Element or a numpy array
            Data element to be denormalized

        Returns
        -------
        element:Element or numpy array
            Denormalized Data
        """

        if self._param is None:
            raise ImplementationError('Normalization parameters not set')

        if not all(self._std):
            return p_data.set_values(self._mean) if isinstance(p_data, Element) else self._mean

        if isinstance(p_data, Element):

            p_data.set_values(np.multiply(p_data.get_values(), 1 / self._param[0]) + (
                    self._param[1] / self._param[0]))

        elif isinstance(p_data, np.ndarray):
            p_data = np.multiply(p_data, 1 / self._param[0]) + \
                     (self._param[1] / self._param[0])
            p_data = np.nan_to_num(p_data)
        else:
            raise ParamError('Wrong datatype provided for denormalization')

        return p_data
