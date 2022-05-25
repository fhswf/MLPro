## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.9 (2021-02-25)

This module provides an environment of customizable Gridworld.
"""

from mlpro.rl.models import *
from mlpro.gt.models import *
import numpy as np
         
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GridWorld (Environment):
    """
    Custom environment of an n-D grid world where the agent 
    has to go to a random target.
    """
    C_NAME          = 'GridWorld'
    C_LATENCY       = timedelta(0,1,0)
    C_INFINITY      = np.finfo(np.float32).max  
    C_REWARD_TYPE   = Reward.C_TYPE_OVERALL
    
## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=True, grid_size=(8,8),
                random_start_position=True,
                random_goal_position=True, max_step=10):
        """
        Parameters:
            p_logging               Boolean switch for logging
            grid_size               Dimension of the grid world
            random_start_position   Randomize start position
            random_goal_position    Randomize goal position
            max_step                Maximum step per episode
        """
        
                
        self.grid_size = np.array(grid_size)
        self.random_start_position = random_start_position
        self.random_goal_position = random_goal_position

        self.agent_pos = np.array([np.random.randint(0,border-1) 
                            if self.random_start_position 
                            else 0 for border in self.grid_size])

        self.goal_pos = np.array([np.random.randint(0,border-1) 
                            if self.random_goal_position 
                            else border-1 for border in self.grid_size])
        self.max_step = max_step
        self.num_step = 0
        
        super().__init__(p_mode=Environment.C_MODE_SIM, p_logging=p_logging)
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
            state_space.add_dim(Dimension( p_name_short=str(i), p_base_set=Dimension.C_BASE_SET_Z, p_boundaries=[0,3]))
        
        for i in range(len(self.grid_size)):
            action_space.add_dim(Dimension( p_name_short=str(i), p_base_set=Dimension.C_BASE_SET_Z, p_boundaries=[-self.grid_size[i], self.grid_size[i]]))

        return state_space, action_space

    
## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        """
        To reset environment
        """
        random.seed(p_seed)
        self.agent_pos = np.array([np.random.randint(0,border-1) 
                            if self.random_start_position 
                            else 0 for border in self.grid_size])

        self.goal_pos = np.array([np.random.randint(0,border) 
                            if self.random_goal_position 
                            else border-1 for border in self.grid_size])
        self.num_step = 0
        self._state = self.get_state()
        

## -------------------------------------------------------------------------------------------------
    def get_state(self):
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
        self.agent_pos += np.array(p_action.get_sorted_values()).astype(int)
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size-1)
        
        self.num_step += 1
        euclidean_distance = np.linalg.norm(self.goal_pos-self.agent_pos)
        if euclidean_distance == 0:
            self._state.set_success(True)
        else:
            self._state.set_success(False)
        self._state = self.get_state()
        return self._state
        

## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old:State, p_state_new:State) -> Reward:
        reward = Reward(self.C_REWARD_TYPE)
        rew = 1
        euclidean_distance = np.linalg.norm(self.goal_pos-self.agent_pos).item()
        if euclidean_distance !=0:
            rew = 1/euclidean_distance
        if self.num_step >= self.max_step:
            rew -= self.max_step
        
        reward.set_overall_reward(rew)
        return reward


## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state:State) -> bool:
        if self.num_step >= self.max_step:
            return True
        elif self.get_success() == True:
            return True

        return False


## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state:State) -> bool:
        return False
        

## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        pass


## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        pass