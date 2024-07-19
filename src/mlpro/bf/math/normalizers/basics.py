## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.math.normalizers
## -- Module  : basics.py
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
## -- 2024-04-30  1.1.0     DA       Refactoring and new class Renormalizable
## -- 2024-05-23  1.2.0     DA       Method Normalizer._set_parameters(): little optimization
## -- 2024-07-12  1.2.1     LSB       Renormalization error
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.1 (2024-07-12)

This module provides base class for Normalizers and normalizer objects including MinMax normalization and
normalization by Z transformation.
"""


from mlpro.bf.exceptions import *
from mlpro.bf.math import Element, Set
import numpy as np
from typing import Union




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Normalizer:
    """
    Base class for normalizers.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self):

        self._param_valid = False
        self._param     = None
        self._param_old = None
        self._param_new = None


## -------------------------------------------------------------------------------------------------
    def _set_parameters(self, p_param):
        """
        Custom method to set the normalization parameters

        Parameters
        ----------
        p_set:Set
            Set related to the elements to be normalized

        Returns
        -------
        boolean:True
            Returns true after setting the parameters
        """
        self._param = p_param.copy()


## -------------------------------------------------------------------------------------------------
    def normalize(self, p_data: Union[Element, np.ndarray]):
        """
        Method to normalize a data (Element/ndarray) element based on MinMax or Z-transformation

        Parameters
        ----------
        p_data:Element or a numpy array
            Data element to be normalized

        Returns
        -------
        element:Element or numpy array
            Normalized Data
        """

        if self._param is None:
            raise ImplementationError('Normalization parameters not set')
        if isinstance(p_data, Element):
            p_data.set_values(np.multiply(p_data.get_values(), self._param[0]) - self._param[1])
        elif isinstance(p_data, np.ndarray):
            p_data = np.multiply(p_data, self._param[0]) - self._param[1]
        else:
            raise ParamError('Wrong data type provided for normalization')
        return p_data


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


## -------------------------------------------------------------------------------------------------
    def renormalize(self, p_data: Union[Element, np.ndarray]):
        """
        Method to denormalize and renormalize an element based on old and current normalization parameters.

        Parameters
        ----------
        p_data:Element or numpy array
            Element to be renormalized.

        Returns
        -------
        renormalized_element:Element or numpy array
            Renormalized Data

        """

        self._set_parameters(self._param_old)
        denormalized_element = self.denormalize(p_data)
        self._set_parameters(self._param_new)
        renormalized_element = self.normalize(denormalized_element)
        return renormalized_element


## -------------------------------------------------------------------------------------------------
    def update_parameters(self, p_data: Union[Set, Element, np.ndarray]):
        """
        Custom method to update normalization parameters.

        Parameters
        ----------
        p_data
            arguments specific to normalization parameters. Check the normalizer objects for specific parameters

        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Renormalizable:
    """
    Property class to add custom renomalization of internally stored data.
    """

## -------------------------------------------------------------------------------------------------
    def renormalize( self, p_normalizer : Normalizer ):
        """
        Custom method to renormalize internally stored data. 

        Parameters
        ----------
        p_normalizer : Normalizier
            Suitable normalizer object to be used for renormalization.
        """

        pass