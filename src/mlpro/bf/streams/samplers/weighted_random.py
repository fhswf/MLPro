## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams.samplers
## -- Module  : weighted_random.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-16  0.0.0     SY       Creation 
## -- 2023-04-16  1.0.0     SY       First version release
## -- 2025-06-06  1.1.0     DA       Refactoring: p_inst -> p_instances
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-06-06)

This module provides a ready-to-use stream sampler class SamplerWeightedRND, in which each instance
is randomly uniformly weighted. Then, it is compared to a pre-defined threshold. If the weight of an
instance is higher than the threshold, then the instance is not omitted. Otherwise, it is omitted.

"""

from mlpro.bf.streams.basics import Sampler, Instance
from mlpro.bf.exceptions import ParamError
import random



# Export list for public API
__all__ = [ 'SamplerWeightedRND' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SamplerWeightedRND(Sampler):
    """
    A ready-to-use class for data streams with random sampler and weighted instance.
    This object can be used in Stream.

    Parameters
    ----------
    p_num_instances : int
        Number of instances. This parameter has no affect in this sampler method. Default = 0.
    p_threshold : float
        Threshold for selection of an instance. This value must be between 0 to 1. Default = 0.5.
    p_seed : int
        Random seeding. Default = 0.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_num_instances:int=0, p_threshold:float=0.5, p_seed:int=0):
        
        super().__init__(p_num_instances=p_num_instances, p_threshold=p_threshold, p_seed=p_seed)
        
        try:
            self._threshold = self._kwargs['p_threshold']
        except:
            raise ParamError('Parameter p_threshold is missing.')
        
        self.reset()
        

## -------------------------------------------------------------------------------------------------
    def reset(self):
        """
        A method to reset the sampler's settings.
        """
        
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
        
        weight_inst = random.uniform(0, 1)

        if weight_inst >= self._threshold:
            return False
        else:
            return True
        
            