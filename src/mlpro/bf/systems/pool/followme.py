## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.systems.pool
## -- Module  : followme.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-10  0.0.0     DA       Initial implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2024-10-10)

This module provides a simple demo system that just cumulates a percentage part of the incoming
action to the inner state.
"""


from mlpro.bf.various import Log
from mlpro.bf.ops import Mode
from mlpro.bf.mt import Task
from mlpro.bf.math import MSpace
from mlpro.bf.systems import State, Action, System




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class FollowMe (System):
    """
    """

    C_NAME      = 'Follow me'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_id=None, 
                  p_name = None, 
                  p_num_dim: int = 1,
                  p_delay: float = 0.5,
                  p_range_max = Task.C_RANGE_NONE, 
                  p_visualize = False, 
                  p_logging=Log.C_LOG_ALL, 
                  **p_kwargs ):
        
        super().__init__( p_id = p_id, 
                          p_name = p_name, 
                          p_range_max = p_range_max, 
                          p_mode = Mode.C_MODE_SIM, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging )
        
        self._action_factor: float = min(1, max(0, 1 - p_delay))
        self._state_space, self._action_space = self._setup_spaces(p_num_dim=p_num_dim)
        

## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self, p_num_dim: int):

        state_space : MSpace = MSpace()
        action_space : MSpace = MSpace()
        
        return state_space, action_space
    

## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None):
        
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action, p_step = None):
        
        raise NotImplementedError