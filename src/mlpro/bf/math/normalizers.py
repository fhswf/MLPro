 ## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.pool
## -- Module  : normalizers.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-09-16  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2022-09-16)
This module provides base class for Normalizers.
"""

from mlpro.bf.math import *
from mlpro.rl.models_sar import *
import numpy



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Normalizer:

    C_TYPE = 'Normalizer'
    C_NAME = ''


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_set:Set):


        self._param = None
        self._old_param = None
        self._new_param = None
        self._set_parameters(p_set)


## -------------------------------------------------------------------------------------------------
    def get_parameters(self):
        return self._param


## -------------------------------------------------------------------------------------------------
    def _set_parameters(self, p_set:Set):


        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def normalize(self, p_element:Element):

        element = self._normalize(p_element)
        return element


## -------------------------------------------------------------------------------------------------
    def _normalize(self, p_element:Element):

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def denormalize(self,p_element:Element):
        element = self._denormalize(p_element)
        return element


## -------------------------------------------------------------------------------------------------
    def _denormalize(self, p_element:Element):

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def renormalize(self, p_element:Element):

        denormalized_element = self.denormalize(p_element)
        renormalized_element = self.normalize(denormalized_element)
        return renormalized_element



## -------------------------------------------------------------------------------------------------
    def _update_param(self, **p_kwargs):

        raise





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerMinMax(Normalizer):


    C_NAME = 'MinMax'


## -------------------------------------------------------------------------------------------------
    def _normalize(self, p_element:Element):

        normalized_element = np.multiply(p_element.get_values(), self._param[0])+self._param[1]
        return normalized_element


## -------------------------------------------------------------------------------------------------
    def _denormalize(self, p_element:Element):

        pass


## -------------------------------------------------------------------------------------------------
    def update_param(self, p_set:Set):

        pass


## -------------------------------------------------------------------------------------------------
    def _set_parameters(self, p_set:Set):

        if p_set is None: raise ValueError('Set not provided')
        a = []
        b = []
        for i in p_set.get_dim_ids():
            min_boundary = p_set.get_dim(i).get_boundaries()[0]
            max_boundary = p_set.get_dim(i).get_boundaries()[1]
            range = max_boundary-min_boundary
            a.append(2/(range))
            b.append(2*min_boundary/(range)+1)

        self._param = np.vstack(([a],[b]))
