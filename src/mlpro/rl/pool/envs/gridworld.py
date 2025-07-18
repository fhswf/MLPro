## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.rl.pool.envs
## -- Module  : gridworld
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-06  0.0.0     WB       Creation
## -- 2021-09-09  1.0.0     WB       Release of first version
## -- 2021-09-11  1.0.1     MRD      Fix compability with mlpro structure
## -- 2021-09-11  1.0.1     MRD      Change Header information to match our new library name
## -- 2021-09-13  1.0.2     WB       Fix on simulate reaction      
## -- 2021-09-30  1.0.3     SY       State-space and action-space improvement     
## -- 2021-10-05  1.0.4     SY       Update following new attributes done and broken in State
## -- 2021-11-15  1.0.5     DA       Refactoring
## -- 2021-12-03  1.0.6     DA       Refactoring
## -- 2021-12-19  1.0.7     DA       Replaced 'done' by 'success'
## -- 2021-12-21  1.0.8     DA       Class GridWorld: renamed method reset() to _reset()
## -- 2022-02-25  1.0.9     SY       Refactoring due to auto generated ID in class Dimension
## -- 2022-09-19  2.0.0     SY       Add discrete action as an option and predefined target
## -- 2022-10-07  2.0.1     SY       Boundaries updates and reward function updates
## -- 2022-10-08  2.0.2     SY       Bug fixing
## -- 2022-11-29  2.0.3     DA       Bug fixing
## -- 2023-04-12  2.0.4     SY       Refactoring 
## -- 2024-10-09  2.0.5     SY       Updating _reset() due to seeding errors
## -- 2025-07-17  2.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.1.0 (2025-07-17) 

This module provides an environment of customizable Gridworld.
"""

from datetime import timedelta

import numpy as np

from mlpro.bf import Log    
from mlpro.bf.math import Dimension, ESpace
from mlpro.bf.systems import State, Action 

from mlpro.rl import Reward, Environment
      


# Export list for public API
__all__ = [ 'GridWorld' ]
    
    
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GridWorld(Environment):
    """
    Custom environment of an n-D grid world where the agent has to go to a random or defined target.
    
    Parameters
    ----------
    p_logging : bool     
        Subspace of an environment that is observed by the policy. Default = Log.C_LOG_ALL.
    p_grid_size : dimension
        Dimension of the grid world (n-D grid world), e.g. (8,8) for 2-D or (8,8,8) for 3-D.
        Default = (8,8)
    p_random_start_position : bool           
        Randomize start position. Default = True.
    p_random_goal_position : bool               
        Randomize goal position. Default = True.
    p_max_step : int
        Maximum step per episode. Default = 50.
    p_action_type : int
        Type of actions, which is either continuous action or discrete action.
        To be noted, discrete action is now limited to 2-d grid world. Default = C_ACTION_TYPE_C.
    p_start_position : dimension           
        To define the starting position, if p_random_start_position is False, e.g. (3,2).
        Default = None.
    p_goal_position : dimension               
        To define the goal positoin, if p_random_goal_position is False, e.g. (5,5).
        Default = None.
    """
    C_NAME                  = 'Grid World'
    C_LATENCY               = timedelta(0,1,0)
    C_INFINITY              = np.finfo(np.float32).max  
    C_REWARD_TYPE           = Reward.C_TYPE_OVERALL
    C_ACTION_TYPE_CONT      = 0
    C_ACTION_TYPE_DISC_2D   = 1
    
    
## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_logging:bool=Log.C_LOG_ALL,
                 p_grid_size=(8,8),
                 p_random_start_position:bool=True,
                 p_random_goal_position:bool=True,
                 p_max_step:int=50,
                 p_action_type:int=C_ACTION_TYPE_CONT,
                 p_visualize=True,
                 p_start_position=None,
                 p_goal_position=None):
        
        self.grid_size = np.array(p_grid_size)
        self.random_start_position = p_random_start_position
        self.random_goal_position = p_random_goal_position
        self.start_position = p_start_position
        self.goal_position  = p_goal_position
            
        self.max_step = p_max_step
        self.action_type = p_action_type
        
        super().__init__(p_mode=Environment.C_MODE_SIM, p_visualize=p_visualize, p_logging=p_logging)
        self._state_space, self._action_space = self._setup_spaces()
        self.reset()


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        return None, None

        
## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self):

        state_space     = ESpace()
        action_space    = ESpace()
        data            = 1

        for size in self.grid_size:
            data *= size
            
        for i in range(data):
            state_space.add_dim(Dimension(p_name_short=str(i),
                                          p_base_set=Dimension.C_BASE_SET_Z,
                                          p_boundaries=[0, 3]))
        
        if self.action_type == self.C_ACTION_TYPE_CONT:
            for i in range(len(self.grid_size)):
                action_space.add_dim(Dimension(p_name_short=str(i),
                                               p_base_set=Dimension.C_BASE_SET_R,
                                               p_boundaries=[-self.grid_size[i], self.grid_size[i]]))
        elif self.action_type == self.C_ACTION_TYPE_DISC_2D:
            action_space.add_dim(Dimension(p_name_short=str(i),
                                           p_base_set=Dimension.C_BASE_SET_Z,
                                           p_boundaries=[0, 3]))

        return state_space, action_space

    
## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        """
        To reset environment
        """
        np.random.seed(p_seed)
        
        if self.random_start_position:
            self.agent_pos = np.array([np.random.randint(0,border-1) 
                                if self.random_start_position 
                                else 0 for border in self.grid_size])
        else:
            if self.start_position is not None:
                self.agent_pos = np.array(self.start_position)
            else:
                raise NotImplementedError('Please define p_start_position or set p_random_start_position to True!')

        if self.random_goal_position:
            self.goal_pos = np.array([np.random.randint(0,border-1) 
                                if self.random_goal_position 
                                else border-1 for border in self.grid_size])
        else:
            if self.goal_position is not None:
                self.goal_pos = np.array(self.goal_position)
            else:
                raise NotImplementedError('Please define p_goal_position or set p_random_goal_position to True!')
                
        self.num_step = 0
        self._state = self.get_all_states()
        

## -------------------------------------------------------------------------------------------------
    def get_all_states(self):
        obs = np.zeros(self.grid_size, dtype=np.float32)
        if np.allclose(self.agent_pos, self.goal_pos):
            obs[tuple(self.agent_pos)] = 3
        else:
            obs[tuple(self.agent_pos)] = 1
            obs[tuple(self.goal_pos)] = 2
        state = State(self._state_space)
        state.set_values(obs.flatten())
        return state
        

## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state:State, p_action:Action) -> State:
        if self.action_type == self.C_ACTION_TYPE_CONT:
            self.agent_pos += np.array(p_action.get_sorted_values()).astype(int)
            self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size-1)
        elif self.action_type == self.C_ACTION_TYPE_DISC_2D:
            action = np.array(p_action.get_sorted_values()).astype(int)
            if action == 0:
                self.agent_pos += np.array((-1,0))
            elif action == 1:
                self.agent_pos += np.array((0,1))
            elif action == 2:
                self.agent_pos += np.array((1,0))
            elif action == 3:
                self.agent_pos += np.array((0,-1))
            self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size-1)
                
        self.num_step += 1
        
        self._state = self.get_all_states()
        return self._state
        

## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old:State, p_state_new:State) -> Reward:
        reward = Reward(self.C_REWARD_TYPE)
        euclidean_distance = np.linalg.norm(self.goal_pos-self.agent_pos).item()
        if euclidean_distance > 0:
            rew = -euclidean_distance
        else:
            rew = 1
        
        reward.set_overall_reward(rew)
        return reward


## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state:State) -> bool:
        euclidean_distance = np.linalg.norm(self.goal_pos-self.agent_pos)
        if euclidean_distance <= 0:
            self._state.set_success(True)
            self._state.set_terminal(True)
            return True
        else:
            self._state.set_success(False)
            return False


## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state:State) -> bool:
        if self.num_step >= self.max_step:
            self._state.set_broken(True)
            self._state.set_terminal(True)
            return True
        else:
            self._state.set_broken(False)
            return False
        

## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        pass


## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        pass