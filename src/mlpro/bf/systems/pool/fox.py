## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.systems.pool
## -- Module  : fox.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-10  0.1.0     DA       Initial implementation
## -- 2024-10-13  0.2.0     DA       Random state jump
## -- 2025-07-18  0.3.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.3.0 (2025-07-18)

This module provides a simple demo system that just cumulates a percentage part of the incoming
action to the inner state.
"""


import random
from datetime import timedelta

import numpy as np

from mlpro.bf.various import Log
from mlpro.bf.ops import Mode
from mlpro.bf.mt import Task
from mlpro.bf.math import Dimension, MSpace, ESpace
from mlpro.bf.systems import State, Action, System



# Export list for public API
__all__ = [ 'Fox' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Fox (System):
    """
    Dummy system mirroring the input with a delay. Just made for validation and basic demonstration.
    """

    C_NAME          = 'Fox'
    C_BOUNDARIES    = [-10,10]
    C_PLOT_ACTIVE   = False

    C_LATENCY       = timedelta( seconds = 1 )

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_id=None, 
                  p_name = C_NAME, 
                  p_num_dim: int = 1,
                  p_delay: float = 0.8,
                  p_thr_jump: float = 0.1,
                  p_latency : timedelta = None,
                  p_range_max = Task.C_RANGE_NONE, 
                  p_visualize = False, 
                  p_logging=Log.C_LOG_ALL ):
        
        super().__init__( p_id = p_id, 
                          p_name = p_name,
                          p_range_max = p_range_max, 
                          p_mode = Mode.C_MODE_SIM, 
                          p_latency = p_latency,
                          p_visualize = False, 
                          p_logging = p_logging )
        
        self._action_factor: float = min(1, max(0, 1 - p_delay))
        self._thr_jump: float = p_thr_jump
        self._state_space, self._action_space = self._setup_spaces(p_num_dim=p_num_dim)
        

## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self, p_num_dim: int):

        state_action_space : MSpace = ESpace()

        for i in range(p_num_dim):
            state_action_space.add_dim( p_dim = Dimension( p_name_short = 'var ' + str(i),
                                                           p_base_set = Dimension.C_BASE_SET_R,
                                                           p_boundaries = self.C_BOUNDARIES ) )
        
        return state_action_space, state_action_space
    

## -------------------------------------------------------------------------------------------------
    def _jump_away(self, p_state: State):
        p_state.set_initial( p_initial=True )
        p_state.values = [random.uniform( self.C_BOUNDARIES[0], self.C_BOUNDARIES[1] ) for _ in range(self.get_state_space().get_num_dim())]
    

## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None):
        random.seed( p_seed )
        new_state = State( p_state_space = self.get_state_space(), p_initial = True )
        self._jump_away( p_state = new_state )
        self._set_state( p_state = new_state )


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action, p_step = None):
        
        agent_id  = p_action.get_agent_ids()[0]
        new_state = State( p_state_space = self.get_state_space())
        new_state.values = np.array(p_state.values) + np.array(p_action.get_elem(p_id=agent_id).get_values()) * self._action_factor

        if np.linalg.norm( new_state.values ) <= self._thr_jump: self._jump_away( p_state=new_state)

        return new_state