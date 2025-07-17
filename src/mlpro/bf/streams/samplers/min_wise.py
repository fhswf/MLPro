## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams.samplers
## -- Module  : min_wise.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-16  0.0.0     SY       Creation 
## -- 2023-04-16  1.0.0     SY       First version release
## -- 2025-06-06  1.1.0     DA       Refactoring: p_inst -> p_instances
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-06-06)

This module provides a ready-to-use stream sampler class SamplerMinWise, in which a set of instances
is classified as one cluster and each of them is weighted with a random uniform value from 0 to 1.
The smallest weighted instance in the cluster is not omitted, while the other is omitted.
This algorithm is proposed by Suman Nath, et al.

"""

from mlpro.bf.streams.basics import Sampler, Instance
from mlpro.bf.exceptions import ParamError
import random


# Export list for public API
__all__ = [ 'SamplerMinWise' ]



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SamplerMinWise(Sampler):
    """
    A ready-to-use class for data streams with min-wise sampler. This object can be used in Stream.

    Parameters
    ----------
    p_num_instances : int
        Number of instances. This parameter has no affect in this sampler method. Default = 0.
    p_cluster_size : int
        Number of instances in a cluster. Default = 10.
    p_seed : int
        Random seeding. Default = 0.
    """

    C_SCIREF_TYPE_PROCEEDINGS   = "Proceedings"
    C_SCIREF_TYPE               = C_SCIREF_TYPE_PROCEEDINGS
    C_SCIREF_AUTHOR             = 'Suman Nath, Phillip B. Gibbons, Srinivasan Seshan, and Zachary R. Anderson'
    C_SCIREF_TITLE              = 'Synopsis Diffusion for Robust Aggregation in Sensor Networks'
    C_SCIREF_YEAR               = '2004'
    C_SCIREF_ISBN               = '1581138792'
    C_SCIREF_PUBLISHER          = 'Association for Computing Machinery'
    C_SCIREF_URL                = 'https://doi.org/10.1145/1031495.1031525'
    C_SCIREF_DOI                = '10.1145/1031495.1031525'
    C_SCIREF_BOOKTITLE          = 'Proceedings of the 2nd International Conference on Embedded Networked Sensor Systems'
    C_SCIREF_PAGES              = '250–262'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_num_instances:int=0, p_cluster_size:float=1, p_seed:int=0):
        
        super().__init__(p_num_instances=p_num_instances, p_cluster_size=p_cluster_size, p_seed=p_seed)
        
        try:
            self._cluster_size = self._kwargs['p_cluster_size']
        except:
            raise ParamError('Parameter p_cluster_size is missing.')
        
        self.reset()
        

## -------------------------------------------------------------------------------------------------
    def reset(self):
        """
        A method to reset the sampler's settings.
        """
        
        self._idx       = 0
        self._cluster   = []
        self._update    = True
        self._min_value = 0
        random.seed(self._kwargs['p_seed'])
        

## -------------------------------------------------------------------------------------------------
    def _omit_instance(self, p_instance : Instance) -> bool:
        """
        A custom method to filter any incoming instances, which is being called by omit_instance()
        method.

        Parameters
        ----------
        p_instance : Instance
            An input instance to be filtered.

        Returns
        -------
        bool
            False means the input instance is not omitted, otherwise True.

        """
        
        if self._idx == self._cluster_size:
            self._update    = True
            self._idx       = 0
            self._cluster   = []

        if self._update:
            for i in range(self._cluster_size):
                self._cluster.append(random.uniform(0,1))
            self._min_value = min(self._cluster)
            self._update    = False
        
        if self._idx == self._cluster.index(self._min_value):
            self._idx += 1
            return False
        else:
            self._idx += 1
            return True