## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.pool.policies
## -- Module  : dummy
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-08  1.0.0     MRD      Creation
## -- 2022-11-02  1.1.0     DA       - Refactoring: methods adapt(), _adapt()
## --                                - Code cleaning
## -- 2022-11-07  1.2.0     DA       Refactoring
## -- 2025-07-17  1.3.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.0 (2025-07-17) 

This module provide Policy Dummy for unittest purpose.
"""

import random

import numpy as np

from mlpro.bf import Log
from mlpro.bf.math import MSpace
from mlpro.bf.systems import State, Action

from mlpro.rl import SARSElement, Policy
from mlpro.rl.pool.sarsbuffer.RandomSARSBuffer import RandomSARSBuffer



# Export list for public API
__all__ = [ 'MyDummyPolicy' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyDummyPolicy (Policy):
    """
    Creates a policy that satisfies mlpro interface.
         
    Parameters
    ----------
     p_observation_space : MSpace     
        Subspace of an environment that is observed by the policy.
    p_action_space : MSpace
        Action space object.
    p_buffer_size : int           
        Size of internal buffer. Default = 1.
    p_batch_size : int
        Batch size. Default = 5
    p_warm_up_step : int
        Warm up step. Default = 10.
    p_ada : bool               
        Boolean switch for adaptivity. Default = True.
    p_visualize : bool
        Boolean switch for env/agent visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default = Log.C_LOG_ALL.
    """

    C_NAME          = 'MyPolicy'
    C_BUFFER_CLS    = RandomSARSBuffer

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_observation_space:MSpace, 
                  p_action_space:MSpace, 
                  p_buffer_size, 
                  p_batch_size=5, 
                  p_warm_up_step=10, 
                  p_ada:bool=True, 
                  p_visualize:bool=False,
                  p_logging=Log.C_LOG_ALL ):

        super().__init__ ( p_observation_space=p_observation_space, 
                           p_action_space=p_action_space, 
                           p_buffer_size=p_buffer_size, 
                           p_ada=p_ada, 
                           p_visualize=p_visualize, 
                           p_logging=p_logging )

        self._state_space   = p_observation_space
        self._action_space  = p_action_space
        self.warm_up_phase = p_warm_up_step
        self.batch_size = p_batch_size
        self.additional_buffer_element = {}
        self.set_id(0)


## -------------------------------------------------------------------------------------------------
    def add_buffer(self, p_buffer_element: SARSElement):
        """
        Intended to save the data to the buffer. By default it save the SARS data.
        
        """
        buffer_element = self._add_additional_buffer(p_buffer_element)
        self._buffer.add_element(buffer_element)


## -------------------------------------------------------------------------------------------------
    def _add_additional_buffer(self, p_buffer_element: SARSElement):
        p_buffer_element.add_value_element(self.additional_buffer_element)
        return p_buffer_element


## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        self._buffer.clear()


## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state: State) -> Action:
        # 1 Create a numpy array for your action values 
        my_action_values = np.zeros(self._action_space.get_num_dim())

        # 2 Computing action values is up to you...
        for d in range(self._action_space.get_num_dim()):
            my_action_values[d] = random.random() 

        # 3 Return an action object with your values
        return Action(self._id, self._action_space, my_action_values)


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_sars_elem:SARSElement) -> bool:
        # 1 Adapting the internal policy is up to you...
        # Add data to buffer
        self.add_buffer(p_sars_elem)

        if len(self._buffer) < self.warm_up_phase:
            self.log(self.C_LOG_TYPE_I, 'Buffer is not full yet, keep collecting data!')
            return False

        sar_data = self._buffer.get_sample(self.batch_size)

        # 2 Only return True if something has been adapted...
        return True