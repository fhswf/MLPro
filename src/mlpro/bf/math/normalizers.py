 ## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.pool
## -- Module  : normalizers.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-09-16  0.0.0     LSB      Creation
## -- 2022-09-20  1.0.0     LSB      Release of first version
## -- 2022-09-23  1.0.1     LSB      Refactoring
## -- 2022-09-26  1.0.2     LSB      Refatoring and reduced custom normalize and denormalize methods
## -- 2022-10-01  1.0.3     LSB      Refactoring and redefining the update parameter method
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2022-09-25)
This module provides base class for Normalizers and normalizer objects including MinMax normalization and 
normalization by Z transformation.
"""

from mlpro.bf.math import *
from mlpro.rl.models_sar import *
import numpy as np
from typing import Union



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Normalizer:
    """
    Base template class for normalizer objects
    """
    C_TYPE = 'Normalizer'


## -------------------------------------------------------------------------------------------------
    def __init__(self):

        self._param = None
        self._param_old = None
        self._param_new = None


## -------------------------------------------------------------------------------------------------
    def _set_parameters(self, p_param):
        """
        custom method to set the normalization parameters

        Parameters
        ----------
        p_set:Set
            Set related to the elements to be normalized

        Returns
        -------
        boolean:True
            Returns true after setting the parameters
        """
        self._param = p_param

## -------------------------------------------------------------------------------------------------
    def normalize(self, p_data:Union[Element, np.ndarray]):
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
            raise ImplementationError('normalization parameters not set')
        if isinstance(p_data, Element):
            normalized_element = Element(p_data.get_related_set())
            normalized_element.set_values(np.multiply(p_data.get_values(), self._param[0]) - self._param[1])
        elif isinstance(p_data, np.ndarray):
            normalized_element = np.multiply(p_data, self._param[0]) - self._param[1]
        else: raise ParamError('wrong data type provided for normalization')
        return normalized_element


## -------------------------------------------------------------------------------------------------
    def denormalize(self, p_data:Union[Element, np.ndarray]):
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
            raise ImplementationError('normalization parameters not set')
        if isinstance(p_data, Element):
            denormalized_element = Element(p_data.get_related_set())
            denormalized_element.set_values(np.multiply(p_data.get_values(), 1 / self._param[0]) + (
                        self._param[1] / self._param[0]))
        elif isinstance(p_data, np.ndarray):
            denormalized_element = np.multiply(p_data, 1 / self._param[0]) + \
                                   (self._param[1] / self._param[0])
        else:
            raise ParamError('wrong datatype provided for denormalization')
        return denormalized_element


## -------------------------------------------------------------------------------------------------
    def renormalize(self, p_data:Union[Element, np.ndarray]):
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
    def update_parameters(self, p_data:Union[Set, Element, np.ndarray]):
        """
        Custom method to update normalization parameters.

        Parameters
        ----------
        p_data
            arguments specific to normalization parameters. Check the normalizer objects for specific parameters

        Returns
        -------
        boolean = True
            Returns true after updating
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerMinMax(Normalizer):
    """
    Class to normalize elements based on MinMax normalization
    """

    C_NAME = 'MinMax'


## -------------------------------------------------------------------------------------------------
    def update_parameters(self, p_set:Set=None, p_boundaries=None):
        """
        custom method to update the normalization parameters

        Parameters
        ----------
        p_set:Set
            Set related to the elements to be normalized
        p_boundaries:ndarray
            array consisting of boundaries related to the dimension of the array

        Returns
        -------
        boolean:True
            Returns true after setting the parameters
        """

        if p_set is None and p_boundaries is None: raise ParamError('Set/boundaries not provided')
        a = []
        b = []
        if p_boundaries is None:
            for i in p_set.get_dim_ids():
                min_boundary = p_set.get_dim(i).get_boundaries()[0]
                max_boundary = p_set.get_dim(i).get_boundaries()[1]
                range = max_boundary-min_boundary
                a.append(2/(range))
                b.append(2*min_boundary/(range)+1)
        if p_set is None:
            for i in p_boundaries:
                p_boundaries.reshape(-1,2)
                min_boundary = p_boundaries[i][0]
                max_boundary = p_boundaries[i][1]
                range = max_boundary-min_boundary
                a.append(2/(range))
                b.append(2*min_boundary/(range)+1)
                np.array([a]).reshape(p_boundaries.shape[0:-1])
                np.array([b]).reshape(p_boundaries.shape[0:-1])
        self._param_old = self._param_new
        self._param_new = np.vstack(([a],[b]))
        self._param = self._param_new
        return True




## -------------------------------------------------------------------------------------------------
class NormalizerZTrans(Normalizer):
    """
    Class for Normalization based on Z transformation
    """
    C_NAME = 'Z-Transformation'


## -------------------------------------------------------------------------------------------------
    def update_parameters(self, p_dataset):
        """
        custom method to update the normalization parameters

        Parameters
        ----------
        p_dataset:numpy array
            Dataset related to the elements to be normalized

        Returns
        -------
        boolean:True
            Returns true after setting the parameters
        """
        std = np.std(p_dataset, axis=0, dtype=np.float64)
        mean = np.mean(p_dataset, axis = 0, dtype=np.float64)

        a = 1/std
        b = mean/std

        self._param_old = self._param_new
        self._param_new = np.vstack(([a], [b]))
        self._param = self._param_new

        return True