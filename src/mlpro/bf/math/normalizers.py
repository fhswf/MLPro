 ## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.pool
## -- Module  : normalizers.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-09-16  0.0.0     LSB      Creation
## -- 2022-09-dd  1.0.0     LSB      Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-09-dd)
This module provides base class for Normalizers.
"""

from mlpro.bf.math import *
from mlpro.rl.models_sar import *
import numpy



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Normalizer:
    """
    Base template class for normalizer objects

    Parameters
    ----------
    p_ser:Set
        Related set/space to the element to be normalized
    """
    C_TYPE = 'Normalizer'
    C_NAME = ''


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_set:Set=None):

        self._param = None
        self._old_param = None
        self._new_param = None
        if p_set is not None:
            self._set_parameters(p_set)


## -------------------------------------------------------------------------------------------------
    def get_parameters(self):
        """
        method to get the normalization parameters
        """
        return self._param


## -------------------------------------------------------------------------------------------------
    def _set_parameters(self, p_set:Set):
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
    def normalize(self, p_element:Element):
        """
        method to normalize element

        Parameters
        ----------
        p_element:Element
            Element to be normalized

        Returns
        -------
        element:Element
            Element to be normalized
        """
        if self._param is None:
            self._set_parameters(p_element.get_related_set())
        element = self._normalize(p_element)
        return element


## -------------------------------------------------------------------------------------------------
    def _normalize(self, p_element:Element):
        """
        Custom method to normalize the element

        Parameters
        ----------
        p_element:Element
            Element to be normalized

        Returns
        -------
        element:Element
            Normalized element
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def denormalize(self,p_element:Element, p_param=None):
        """
        Method to denormalize the normalized elements.

        Parameters
        ----------
        p_element:Element
            Element to be denormalized
        p_param  -Optional
            Parameters to be normalized

        Returns
        -------
        element:Element
            Returns denormzalized element
        """
        if self._param is None:
            self._set_parameters(p_element.get_related_set())
        element = self._denormalize(p_element, p_param)
        return element


## -------------------------------------------------------------------------------------------------
    def _denormalize(self, p_element:Element, p_param = None):
        """
        Custom method to denormalize an element

        Parameters
        ----------
        p_element:Element
            Element to be denormalized
        p_param  -Optional
            Parameters for denormalization

        Returns
        -------
        denormalized_element:Element
            Denormalized element
        """
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def renormalize(self, p_element:Element):
        """
        Method to denormalize and renormalize an element based on old and current normalization parameters.

        Parameters
        ----------
        p_element:Element
            Element to be renormalized.

        Returns
        -------
        renormalized_elemet:Element
            Renormalized element

        """

        denormalized_element = self.denormalize(p_element, p_param = self._old_param)
        renormalized_element = self.normalize(denormalized_element)
        return renormalized_element



## -------------------------------------------------------------------------------------------------
    def _update_param(self, **p_kwargs):
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
    def _normalize(self, p_element:Element):
        """
        Custom method to normalize the element

        Parameters
        ----------
        p_element:Element
                Element to be normalized

        Returns
        -------
        element:Element
            Normalized element
        """

        normalized_element = np.multiply(p_element.get_values(), self._param[0])-self._param[1]
        return normalized_element


## -------------------------------------------------------------------------------------------------
    def _denormalize(self, p_element:Element, p_param = None):
        """
        Method to denormalize data by inverse minmax.

        Parameters
        ----------
        p_element:Element
            Element to be normalized
        p_param   - Optional
            Parameters for de-normalization
        Returns
        -------
        denormalized_event:Element
            Denormalized event

        """
        if p_param is None:
            p_param = self._param

        denormalized_element = np.multiply(p_element.get_values(), (1/p_param[0]))+p_param[2]
        return denormalized_element


## -------------------------------------------------------------------------------------------------
    def update_param(self, p_element:Element):
        """
        Custom method to update the parameters

        Parameters
        ----------
        p_element:Element
            New element with cahnged boundary data.


        """

        self._old_param = self._param
        self._set_parameters(p_element.get_related_set())
        return True


## -------------------------------------------------------------------------------------------------
    def _set_parameters(self, p_set:Set):
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

        if p_set is None: raise ValueError('Set not provided')
        a = []
        b = []
        c = []
        for i in p_set.get_dim_ids():
            min_boundary = p_set.get_dim(i).get_boundaries()[0]
            max_boundary = p_set.get_dim(i).get_boundaries()[1]
            range = max_boundary-min_boundary
            a.append(2/(range))
            b.append(2*min_boundary/(range)+1)
            c.append(min_boundary+range/2)

        self._param = np.vstack(([a],[b],[c]))

        return True




## -------------------------------------------------------------------------------------------------
class NormalizerZTrans(Normalizer):

    C_NAME = 'Z-Transformation'


## -------------------------------------------------------------------------------------------------
    def _normalize(self, p_element:Element):

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _denormalize(self, p_element:Element, p_param=None):

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def update_param(self, p_data):

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _set_parameters(self, p_set:Set):
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
        return True