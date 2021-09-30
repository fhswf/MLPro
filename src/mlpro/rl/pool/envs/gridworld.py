## -----------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : gridworld
## -----------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.  Auth.  Description
## -- 2021-09-06  0.00  WB     Creation
## -- 2021-09-09  1.00  WB     Release of first version
## -- 2021-09-11  1.01  MRD    Fix compability with mlpro structure
## -- 2021-09-11  1.01  MRD    Change Header information to match our new library name
## -- 2021-09-13  1.02  WB     Fix on simulate reaction      
## -- 2021-09-30  1.03  SY     State-space and action-space improvement     
## -----------------------------------------------------------------------------

"""
Ver. 1.03 (2021-09-30)

This module provides an environment of customizable Gridworld.
"""

from mlpro.rl.models import *
from mlpro.gt.models import *
import numpy as np
         
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GridWorld(Environment):
    """
    Custom environment of an n-D grid world where the agent 
    has to go to a random target.
    """
    C_NAME      = 'GridWorld'
    C_LATENCY   = timedelta(0,1,0)
    C_INFINITY  = np.finfo(np.float32).max  
    
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
        
        super(GridWorld, self).__init__(p_mode=Environment.C_MODE_SIM,
                                        p_logging=p_logging)

    def _setup_spaces(self):
        data = 1
        for size in self.grid_size:
            data *= size
            
        for i in range(data):
            self._state_space.add_dim(Dimension(i, str(i), p_base_set='Z',
                                                p_boundaries=[0,3]))
        
        for i in range(len(self.grid_size)):
            self._action_space.add_dim(Dimension(i, str(i), p_base_set='Z',
                                                 p_boundaries=[-self.grid_size[i], self.grid_size[i]]))
    
    def reset(self) -> None:
        """
        To reset environment
        """
        self.agent_pos = np.array([np.random.randint(0,border-1) 
                            if self.random_start_position 
                            else 0 for border in self.grid_size])

        self.goal_pos = np.array([np.random.randint(0,border) 
                            if self.random_goal_position 
                            else border-1 for border in self.grid_size])
        self.num_step = 0
        self.state = self.get_state()
        
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
        
    def _simulate_reaction(self, p_action: Action) -> None:
        self.agent_pos += np.array(p_action.get_sorted_values()).astype(np.int)
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size-1)
        
        self.num_step += 1
        euclidean_distance = np.linalg.norm(self.goal_pos-self.agent_pos)
        if euclidean_distance == 0:
            self.done = True
        else:
            self.done = False
        self.state = self.get_state()
        
    def compute_reward(self):
        reward = Reward(Reward.C_TYPE_OVERALL)
        rew = 1
        euclidean_distance = np.linalg.norm(self.goal_pos-self.agent_pos)
        if euclidean_distance !=0:
            rew = 1/euclidean_distance
        if self.num_step >= self.max_step:
            rew -= self.max_step
        
        reward.set_overall_reward(rew)
        return reward
        
    def _evaluate_state(self):
        if self.num_step >= self.max_step:
            self.goal_achievement   = 0.0
            self.done               = True
        elif self.done == True:
            self.goal_achievement   = 1.0
            
    
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------









