## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro.wrappers
## -- Module  : rl
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-08-27  0.0.0     SY       Creation
## -- 2021-08-27  1.0.0     SY       Release of first version
## --                                New classes WrEnvPZoo
## -- 2021-09-11  1.0.0     MRD      Change Header information to match our new library name
## -- 2021-09-23  1.1.0     SY       Update WrEnvGym to solve big data issues
## --                                WrEnvPZoo is ready to use
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2021-09-23)

This module provides wrapper classes for reinforcement learning tasks.
"""


import gym
import numpy as np
from typing import List
from time import sleep
from mlpro.bf.various import *
from mlpro.bf.math import *
from mlpro.rl.models import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrEnvGym(Environment):
    """
    This class is a ready to use wrapper class for OpenAI Gym environments. 
    Objects of this type can be treated as an environment object. Encapsulated 
    gym environment must be compatible to class gym.Env.
    """

    C_TYPE        = 'OpenAI Gym Env'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_gym_env, p_state_space:MSpace=None, p_action_space:MSpace=None, p_logging=True):
        """
        Parameters:
            p_gym_env       Gym environment object
            p_state_space   Optional external state space object that meets the
                            state space of the gym environment
            p_action_space  Optional external action space object that meets the
                            state space of the gym environment
            p_logging       Switch for logging
        """

        self._gym_env     = p_gym_env
        self.C_NAME       = 'Env "' + self._gym_env.spec.id + '"'
        Environment.__init__(self, Environment.C_MODE_SIM, None, p_logging)
        
        if p_state_space is not None: 
            self._state_space = p_state_space
        else:
            self._state_space = self._recognize_space(self._gym_env.observation_space, "observation")
        
        if p_action_space is not None: 
            self._action_space = p_action_space
        else:
            self._action_space = self._recognize_space(self._gym_env.action_space, "action")

        self.reset()


## -------------------------------------------------------------------------------------------------
    def __del__(self):
        self._gym_env.close()
        self.log(self.C_LOG_TYPE_I, 'Closed')


## -------------------------------------------------------------------------------------------------
    def _recognize_space(self, p_gym_space, dict_name) -> ESpace:
        space = ESpace()
        
        if len(p_gym_space.shape) == 0:
            space.add_dim(Dimension(p_id=0,p_name_short='0', p_boundaries=[p_gym_space.n]))
        else:
            for d in range(p_gym_space.shape[0]):
                space.add_dim(Dimension(p_id=d, p_name_short=str(d), p_boundaries=[p_gym_space.low[d], p_gym_space.high[d]]))
        
        return space


## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self):
        pass


## -------------------------------------------------------------------------------------------------
    def reset(self):
        self.log(self.C_LOG_TYPE_I, 'Reset')

        # 1 Reset Gym environment and determine initial state
        observation = self._gym_env.reset()
        obs         = DataObject(observation)

        # 2 Create state object from Gym observation
        state   = State(self._state_space)
        state.set_values(obs.get_data())
        self._set_state(state)


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_action:Action):

        # 1 Convert action to Gym syntax
        action_sorted = p_action.get_sorted_values()
        dtype         = self._gym_env.action_space.dtype

        if ( dtype == np.int32 ) or ( dtype == np.int64 ):
            action_sorted = action_sorted.round(0)

        if action_sorted.size == 1:
            action_gym = action_sorted.astype(self._gym_env.action_space.dtype)[0]
        else:
            action_gym = action_sorted.astype(self._gym_env.action_space.dtype)

        # 2 Process step of Gym environment
        try:
            observation, reward_gym, self.done, info = self._gym_env.step(action_gym)
        except:
            observation, reward_gym, self.done, info = self._gym_env.step(np.atleast_1d(action_gym))
        obs     = DataObject(observation)

        # 3 Create state object from Gym observation
        state   = State(self._state_space)
        state.set_values(obs.get_data())
        self._set_state(state)

        # 4 Create and store reward object
        self.reward = Reward(Reward.C_TYPE_OVERALL)
        self.reward.set_overall_reward(reward_gym)


## -------------------------------------------------------------------------------------------------
    def _evaluate_state(self):
        if self.done:
            self.goal_achievement = 1.0
        else:
            self.goal_achievement = 0.0


## -------------------------------------------------------------------------------------------------
    def compute_reward(self) -> Reward:
        return self.reward


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        self._gym_env.render()


## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        self._gym_env.render()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrEnvPZoo(Environment):
    """
    This class is a ready to use wrapper class for Petting Zoo environments. 
    Objects of this type can be treated as an environment object. Encapsulated 
    petting zoo environment must be compatible to class pettingzoo.env.
    """

    C_TYPE        = 'Petting Zoo Env'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_zoo_env, p_state_space:MSpace=None, p_action_space:MSpace=None, p_logging=True):
        """
        Parameters:
            p_pzoo_env      Petting Zoo environment object
            p_state_space   Optional external state space object that meets the
                            state space of the gym environment
            p_action_space  Optional external action space object that meets the
                            state space of the gym environment
            p_logging       Switch for logging
        """

        self._zoo_env     = p_zoo_env
        self.C_NAME       = 'Env "' + self._zoo_env.metadata['name'] + '"'
        Environment.__init__(self, Environment.C_MODE_SIM, None, p_logging)
        
        if p_state_space is not None: 
            self._state_space = p_state_space
        else:
            self._state_space = self._recognize_space(self._zoo_env.observation_spaces, "observation")
        
        if p_action_space is not None: 
            self._action_space = p_action_space
        else:
            self._action_space = self._recognize_space(self._zoo_env.action_spaces, "action")

        self.reset()


## -------------------------------------------------------------------------------------------------
    def __del__(self):
        self._zoo_env.close()
        self.log(self.C_LOG_TYPE_I, 'Closed')


## -------------------------------------------------------------------------------------------------
    def _recognize_space(self, p_zoo_space, dict_name) -> ESpace:
        space = ESpace()
        id_ = 0
        
        if dict_name == "observation":
            space.add_dim(Dimension(p_id=0,p_name_short='0', p_base_set='DO'))
        elif dict_name == "action":
            for k in p_zoo_space:
                space.add_dim(Dimension(p_id=id_,p_name_short=k, p_base_set='DO'))
                id_ += 1
                
        return space

## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self):
        pass


## -------------------------------------------------------------------------------------------------
    def reset(self):
        self.log(self.C_LOG_TYPE_I, 'Reset')
        
        # 1 Reset Zoo environment and determine initial state
        self._zoo_env.reset()
        observation, _, _, _ = self._zoo_env.last()
        obs     = DataObject(observation)
        
        # 2 Create state object from Zoo observation
        state   = State(self._state_space)
        if isinstance(observation, dict):
            state.set_values(obs.get_data()['observation'])
        else:
            state.set_values(obs.get_data())
        self._set_state(state)


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_action:Action):

        # 1 Convert action to Zoo syntax
        action_sorted = p_action.get_sorted_values()
        agent_num = 0
        
        for k in self._zoo_env.action_spaces:
            dtype = self._zoo_env.action_spaces[k].dtype
    
            if ( dtype == np.int32 ) or ( dtype == np.int64 ):
                action_sorted_agent = action_sorted[agent_num].round(0)
            else:
                action_sorted_agent = action_sorted[agent_num]
    
            action_zoo = action_sorted_agent.astype(self._zoo_env.action_spaces[k].dtype)
            
        # 2 Process step of Zoo environment that automatically switches control to the next agent.
            observation, reward_zoo, self.done, info = self._zoo_env.last()
            obs     = DataObject(observation)
            
            if self.done:
                self._zoo_env.step(None)
            else:
                try:
                    self._zoo_env.step(action_zoo)
                except:
                    self._zoo_env.step(np.atleast_1d(action_zoo))
            agent_num += 1

        # 3 Create state object from Zoo observation
            state = State(self._state_space)
            if isinstance(observation, dict):
                state.set_values(obs.get_data()['observation'])
            else:
                state.set_values(obs.get_data())
            self._set_state(state)

        # 4 Create and store reward object
        self.reward = Reward(Reward.C_TYPE_OVERALL)
        self.reward.set_overall_reward(reward_zoo)


## -------------------------------------------------------------------------------------------------
    def _evaluate_state(self):
        if self.done:
            self.goal_achievement = 1.0
        else:
            self.goal_achievement = 0.0


## -------------------------------------------------------------------------------------------------
    def compute_reward(self) -> Reward:
        return self.reward


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        self._zoo_env.render()


## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        self._zoo_env.render()





