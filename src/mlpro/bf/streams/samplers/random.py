## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams.samplers
## -- Module  : random.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-10  0.0.0     SY       Creation 
## -- 2023-04-14  1.0.0     SY       First version release
## -- 2025-06-06  1.1.0     DA       Refactoring: p_inst -> p_instances
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-06-06)

This module provides a ready-to-use stream sampler class SamplerRND.

"""

from mlpro.bf.streams.basics import Sampler, Instance
from mlpro.bf.exceptions import ParamError
import random



# Export list for public API
__all__ = [ 'SamplerRND' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SamplerRND(Sampler):
    """
    A ready-to-use class for data streams with random sampler. This object can be used in Stream.

    Parameters
    ----------
    p_num_instances : int
        Number of instances. This parameter has no affect in this sampler method. Default = 0.
    p_max_step_rate : int
        Maximum step rate parameter for non time series data streams. Default = 5.
    p_seed : int
        Random seeding. Default = 0.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_num_instances:int=0, p_max_step_rate:int=5, p_seed:int=0):
        
        super().__init__(p_num_instances=p_num_instances, p_max_step_rate=p_max_step_rate, p_seed=p_seed)
        
        try:
            self._max_step_rate = self._kwargs['p_max_step_rate']
        except:
            raise ParamError('Parameter p_max_step_rate is missing.')
        
        self.reset()
        

## -------------------------------------------------------------------------------------------------
    def reset(self):
        """
        A method to reset the sampler's settings.
        """
        
        self._idx      = 0
        self._nxt_idx  = None
        random.seed(self._kwargs['p_seed'])
        

## -------------------------------------------------------------------------------------------------
    def _omit_instance(self, p_instance:Instance) -> bool:
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
        
        if self._nxt_idx is None:
            self._nxt_idx = random.randint(1, self._max_step_rate)
        if self._idx < self._nxt_idx:
            self._idx += 1
            return True
        else:
            self._idx = 0
            self._nxt_idx = None
            return False
        
            