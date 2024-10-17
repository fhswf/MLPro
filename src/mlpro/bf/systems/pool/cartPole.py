## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.systems.pool
## -- Module  : fox.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-10  0.1.0     DA       Initial implementation
## -- 2024-10-13  0.2.0     DA       Random state jump
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2024-10-13)

This module provides a simple demo system that just cumulates a percentage part of the incoming
action to the inner state.
"""

import numpy as np
from mlpro.bf.various import Log
from mlpro.bf.ops import Mode
from mlpro.bf.mt import Task
from mlpro.bf.math import Dimension, MSpace, ESpace
from mlpro.bf.systems import State, Action, System
import gymnasium as gym
import math




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CartPole (System):
    """
    ...
    """

    C_NAME          = 'Cart Pole'
    C_BOUNDARIES    = [0,1]

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_id=None, 
                  p_name = None, 
                  p_num_dim: int = 1,
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
        
        
        self._env = gym.make('CartPole-v1', render_mode="human")
        self._observation=None
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
    def _radians_to_degrees(self, p_state_value: float):
        p_degree = p_state_value * (180 / math.pi)
        return p_degree
        
    

## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None):
        self._observation = self._env.reset()[0]




## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action, p_step = None):

        self._env.render() 

        p_action_value = p_action.get_feature_data().get_values()[0]

        
        if p_action_value > 0:
            p_action_value = 1  # move right
        else:
            p_action_value = 0  # move left


        # Führe die Aktion in der Umgebung aus
        self._observation,*_ = self._env.step(p_action_value)        
       
        #create new state 
        new_state = State( p_state_space = self.get_state_space())
        #calculate angle of the pendelum 
        new_state.values = self._radians_to_degrees(self._observation[2])      

        return new_state