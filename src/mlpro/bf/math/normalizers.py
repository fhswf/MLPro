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
    C_NAME = ''


## -------------------------------------------------------------------------------------------------
    def __init__(self):

        self._param = None
        self._param_old = None


## -------------------------------------------------------------------------------------------------
    def get_parameters(self):
        """
        method to get the normalization parameters
        """
        return self._param


## -------------------------------------------------------------------------------------------------
    def set_parameters(self, p_data):
        """
        method to set the normalization parameters

        Parameters
        ----------
        p_data
            Data for setting the normalization parameters
        """
        self._set_parameters(p_data)


## -------------------------------------------------------------------------------------------------
    def _set_parameters(self, p_data:Union[Set, np.ndarray]):
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

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def normalize(self, p_element:Union[Element, np.ndarray], p_param=None):
        """
        method to normalize element

        Parameters
        ----------
        p_element:Element or a numpy array
            Element to be normalized

        Returns
        -------
        element:Element or numpy array
            Normalized element
        """

        if self._param is None:
            raise ImplementationError('normalization parameters not set')
        # element = self._normalize(p_element, p_param = self._param)
        p_param = self._param
        if isinstance(p_element, Element):
            normalized_element = Element(p_element.get_related_set())
            normalized_element.set_values(np.multiply(p_element.get_values(), p_param[0]) - p_param[1])
        elif isinstance(p_element, np.ndarray):
            normalized_element = np.multiply(p_element, p_param[0]) - p_param[1]
        else: raise ParamError('wrong data type provided for normalization')
        return normalized_element


## -------------------------------------------------------------------------------------------------
    def denormalize(self, p_element:Union[Element, np.ndarray], p_param=None):
        """
        Method to denormalize the normalized elements.

        Parameters
        ----------
        p_element:Element or a numpy array
            Element to be denormalized
        p_param  -Optional
            Parameters to be normalized

        Returns
        -------
        element:Element or numpy array
            Denormalized element
        """
        if self._param is None:
            raise ImplementationError('normalization parameters not set')
        if p_param is None:
            p_param = self._param
        if isinstance(p_element, Element):
            denormalized_element = Element(p_element.get_related_set())
            denormalized_element.set_values(np.multiply(p_element.get_values(), 1 / p_param[0]) + (
                        p_param[1] / p_param[0]))
        elif isinstance(p_element, np.ndarray):
            denormalized_element = np.multiply(p_element, 1 / p_param[0]) + \
                                   (p_param[1] / p_param[0])
        else:
            raise ParamError('wrong datatype provided for denormalization')
        return denormalized_element


## -------------------------------------------------------------------------------------------------
    def renormalize(self, p_data:Union[Element, np.ndarray]):
        """
        Method to denormalize and renormalize an element based on old and current normalization parameters.

        Parameters
        ----------
        p_element:Element or numpy array
            Element to be renormalized.

        Returns
        -------
        renormalized_elemet:Element or numpy array
            Renormalized element

        """

        denormalized_element = self.denormalize(p_data, p_param = self._param_old)
        renormalized_element = self.normalize(denormalized_element)
        return renormalized_element



## -------------------------------------------------------------------------------------------------
    def _update_param(self, p_data:Union[Set, Element, np.ndarray]):
        """
        Custom method to update normalization parameters.

        Parameters
        ----------
        p_kwargs
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
    def update_param(self, p_element:Element):
        """
        Custom method to update the parameters

        Parameters
        ----------
        p_element:Element
            New element with changed boundary data.

        Returns
        -------
        boolean:True
            returns True after parameter update
        """

        self._old_param = self._param
        self._set_parameters(p_element.get_related_set())
        return True


## -------------------------------------------------------------------------------------------------
    def _set_parameters(self, p_set:Set=None, p_boundaries=None):
        """
        custom method to set the normalization parameters

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
        self._param = np.vstack(([a],[b]))

        return True




## -------------------------------------------------------------------------------------------------
class NormalizerZTrans(Normalizer):
    """
    Class for Normalization based on Z transformation
    """
    C_NAME = 'Z-Transformation'


## -------------------------------------------------------------------------------------------------
    def update_param(self, p_dataset):
        """
        Custom method to update normalization parameters.

        Parameters
        ----------
        p_dataset: numpy array
            Data to be normalized

        Returns
        -------
        boolean = True
            Returns true after updating
        """
        self._old_param = self._param
        self._set_parameters(p_dataset)
        return True


## -------------------------------------------------------------------------------------------------
    def _set_parameters(self, p_dataset):
        """
        custom method to set the normalization parameters

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


        self._param = np.vstack(([a],[b]))

        return True