## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.streams.samplers
## -- Module  : random.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-10  0.0.0     SY       Creation 
## -- 2023-04-10  1.0.0     SY       First version release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-04-10)

This module provides a ready-to-use stream sampler class SamplerRND.

"""

from mlpro.bf.streams.models import Sampler, Instance
import random
from datetime import timedelta





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SamplerRND(Sampler):
    """
    A ready-to-use class for data streams with random sampler. This object can be used in Stream.

    Parameters
    ----------
    p_num_instances : int
        number of instances.
    p_max_step_time : int
        Maximum delta time parameter for time series data streams in seconds.
    p_max_step_rate : int
        Maximum step rate parameter for non time series data streams.
    p_seed : int
        Random seeding.
    """

    C_TYPE          = 'Random Sampler'


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_num_instances:int, **p_kwargs):
        
        super().__init__(p_num_instances=p_num_instances, p_kwargs=p_kwargs)
        
        try:
            self._max_step_time = p_kwargs['p_max_step_time']
        except:
            pass
        
        try:
            self._max_step_rate = p_kwargs['p_max_step_rate']
        except:
            pass
        
        try:
            self._seed  = p_kwargs['p_seed']
        except:
            pass
        
        self.reset()
        


## -------------------------------------------------------------------------------------------------
    def reset(self):
        """
        A method to reset the sampler's settings.
        """
        
        self._idx      = 0
        self._nxt_idx  = None
        self._nxt_time = None
        try:
            random.seed(self._seed)
        except:
            pass
        

## -------------------------------------------------------------------------------------------------
    def _filter_instance(self, p_inst:Instance) -> bool:
        """
        A custom method to filter any incoming instances, which is being called by filter_intance().

        Parameters
        ----------
        p_inst : Instance
            An input instance to be filtered.

        Returns
        -------
        bool
            True means utilize the input instance, otherwise False.

        """
        
        # For time-series data
        if p_inst._time_stamp is not None:
            if self._nxt_time is None:
                self._nxt_time = p_inst._time_stamp + timedelta( seconds=random.randint(1, self._max_step_time) )
            if p_inst._time_stamp < self._nxt_time:
                return False
            else:
                self._nxt_time = p_inst._time_stamp + timedelta( seconds=random.randint(1, self._max_step_time) )
                return True
        
        # For non time-series data
        else:
            if self._nxt_idx is None:
                self._nxt_idx = random.randint(self._idx, self._max_step_rate)
            if self._idx < self._nxt_idx:
                self._idx += 1
                return False
            else:
                self._idx += 1
                self._nxt_idx = random.randint(self._idx, self._idx+self._max_step_rate)
                return True
        
            