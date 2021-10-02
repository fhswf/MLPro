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
## -- 2021-09-24  1.1.1     MRD      Change the gym wrapper _recognize_space() function to seperate
## --                                between discrete space and continuous space
## -- 2021-09-28  1.1.2     SY       WrEnvGym, WrEnvPZoo: implementation of method get_cycle_limits()
## -- 2021-09-29  1.1.3     SY       Change name: WrEnvGym to WrEnvGYM2MLPro, WrEnvPZoo to WrEnvPZOO2MLPro
## -- 2021-09-30  1.2.0     SY       New classes: WrEnvMLPro2GYM
## -- 2021-10-02  1.3.0     SY       New classes: WrEnvMLPro2PZoo, update _recognize_space() in WrEnvGYM2MLPro
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.0 (2021-10-02)

This module provides wrapper classes for reinforcement learning tasks.
"""


import gym
from gym import error, spaces, utils
import numpy as np
from typing import List
from time import sleep
from mlpro.bf.various import *
from mlpro.bf.math import *
from mlpro.rl.models import *
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrEnvGYM2MLPro(Environment):
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
            shape_dim = len(p_gym_space.shape)
            for i in range(shape_dim):
                for d in range(p_gym_space.shape[i]):
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
    def get_cycle_limit(self):
        return self._gym_env._max_episode_steps





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrEnvPZOO2MLPro(Environment):
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


## -------------------------------------------------------------------------------------------------
    def get_cycle_limit(self):
        try:
            return self._zoo_env.env.env.max_cycles
        except:
            return self.C_CYCLE_LIMIT




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrEnvMLPro2GYM(gym.Env):
    """
    This class is a ready to use wrapper class for MLPro to OpenAI Gym environments. 
    Objects of this type can be treated as an gym.Env object. Encapsulated 
    MLPro environment must be compatible to class Environment.
    """

    C_TYPE        = 'MLPro to Gym Env'
    metadata      = {'render.modes': ['human']}

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_mlpro_env, p_state_space:MSpace=None, p_action_space:MSpace=None):
        """
        Parameters:
            p_mlpro_env     MLPro's Environment object
            p_state_space   Optional external state space object that meets the
                            state space of the MLPro environment
            p_action_space  Optional external action space object that meets the
                            state space of the MLPro environment
        """

        self._mlpro_env             = p_mlpro_env
        
        if p_state_space is not None: 
            self.observation_space  = p_state_space
        else:
            self.observation_space  = self._recognize_space(self._mlpro_env._state_space)
        
        if p_action_space is not None: 
            self.action_space       = p_action_space
        else:
            self.action_space       = self._recognize_space(self._mlpro_env._action_space)
        
        self.first_refresh          = True
        self.reset()
        

## -------------------------------------------------------------------------------------------------
    def _recognize_space(self, p_mlpro_space):
        _shape      = p_mlpro_space.get_num_dim()
        ids         = p_mlpro_space.get_dim_ids()[0]
        _low        = p_mlpro_space.get_dim(ids).get_boundaries()[0]
        _high       = p_mlpro_space.get_dim(ids).get_boundaries()[1]
        set_base    = p_mlpro_space.get_dim(ids).get_base_set()
        
        if set_base == 'N' or set_base == 'Z':
            space = spaces.Box(low=_low, high=_high, shape=(_shape,), dtype=np.int)
        else:
            space = spaces.Box(low=_low, high=_high, shape=(_shape,), dtype=np.float32)
            
        return space


## -------------------------------------------------------------------------------------------------
    def step(self, action):
        _action     = Action()
        _act_set    = Set()
        idx         = self._mlpro_env._action_space.get_num_dim()
        for i in range(idx):
            _act_set.add_dim(Dimension(i,'action_'+str(i)))
        _act_elem   = Element(_act_set)
        for i in range(idx):
            _act_elem.set_value(i, action[i].item())
        _action.add_elem('0', _act_elem)
        
        self._mlpro_env.process_action(_action)
        reward          = self._mlpro_env.compute_reward()
        self._mlpro_env._evaluate_state
        
        return self._mlpro_env.get_state().get_values(), reward.get_overall_reward(), self._mlpro_env.done, {}
    

## -------------------------------------------------------------------------------------------------
    def reset(self):
        self._mlpro_env.reset()
        return self._mlpro_env.get_state().get_values()
    

## -------------------------------------------------------------------------------------------------
    def render(self, mode='human'):
        try:
            if self.first_refresh:
                self._mlpro_env.init_plot()
                self.first_refresh = False
            else:
                self._mlpro_env.update_plot()
            return True
        except:
            return False


## -------------------------------------------------------------------------------------------------
    def close(self):
        self._mlpro_env.__del__()






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrEnvMLPro2PZoo():
    """
    This class is a ready to use wrapper class for MLPro to PettingZoo environments. 
    Objects of this type can be treated as an AECEnv object. Encapsulated 
    MLPro environment must be compatible to class Environment.
    To be noted, this wrapper is not capable for parallel environment yet.
    """

    C_TYPE        = 'MLPro to PZoo Env'


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_mlpro_env, p_num_agents, p_state_space:MSpace=None, p_action_space:MSpace=None):
        """
        Parameters:
            p_mlpro_env     MLPro's Environment object
            p_num_agents    Number of Agents
            p_state_space   Optional external state space object that meets the
                            state space of the MLPro environment
            p_action_space  Optional external action space object that meets the
                            state space of the MLPro environment
        """
        
        self.pzoo_env   = self.raw_env(p_mlpro_env, p_num_agents, p_state_space, p_action_space)
        self.pzoo_env   = wrappers.CaptureStdoutWrapper(self.pzoo_env)
        self.pzoo_env   = wrappers.OrderEnforcingWrapper(self.pzoo_env)


## -------------------------------------------------------------------------------------------------
    class raw_env(AECEnv):
        metadata = {'render.modes': ['human'], "name": "pzoo_custom"}

## -------------------------------------------------------------------------------------------------
        def __init__(self, p_mlpro_env, p_num_agents, p_state_space:MSpace=None, p_action_space:MSpace=None):
            self._mlpro_env             = p_mlpro_env
            self.possible_agents        = ["agent_" + str(r) for r in range(p_num_agents)]
            self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
            
            if p_state_space is not None: 
                self.observation_spaces = p_state_space
            else:
                self.observation_spaces = self._recognize_space(self._mlpro_env._state_space)
            
            if p_action_space is not None: 
                self.action_spaces      = p_action_space
            else:
                self.action_spaces      = self._recognize_space(self._mlpro_env._action_space)
            
            self.first_refresh          = True
            self.reset()
        

## -------------------------------------------------------------------------------------------------
        def _recognize_space(self, p_mlpro_space):
            _shape          = p_mlpro_space.get_num_dim()
            ids             = p_mlpro_space.get_dim_ids()[0]
            _low            = p_mlpro_space.get_dim(ids).get_boundaries()[0]
            _high           = p_mlpro_space.get_dim(ids).get_boundaries()[1]
            set_base        = p_mlpro_space.get_dim(ids).get_base_set()
            
            if set_base == 'N' or set_base == 'Z':
                space       = spaces.Box(low=_low, high=_high, shape=(_shape,), dtype=np.int)
            else:
                space       = spaces.Box(low=_low, high=_high, shape=(_shape,), dtype=np.float32)
                
            setup_space     = {agent: space for agent in self.possible_agents}
                
            return setup_space


## -------------------------------------------------------------------------------------------------
        def step(self, action):
            if self.dones[self.agent_selection]:
                return self._was_done_step(action)
            
            agent = self.agent_selection
            self._cumulative_rewards[agent] = 0
            
            if agent == "agent_0":
                self.action_set = []
            self.action_set.append(action[self.agent_selection.index(agent)])
            
            if agent == self.possible_agents[-1]:
                _action     = Action()
                _act_set    = Set()
                idx         = self._mlpro_env._action_space.get_num_dim()
                for i in range(idx):
                    _act_set.add_dim(Dimension(i,'action_'+str(i)))
                _act_elem   = Element(_act_set)
                for i in range(idx):
                    _act_elem.set_value(i, self.action_set[i])
                _action.add_elem('0', _act_elem)
                
                self._mlpro_env.process_action(_action)
                self._mlpro_env._evaluate_state
                
            self.rewards[agent] = self._mlpro_env.compute_reward().get_agent_reward(agent)
            
            if self._mlpro_env.done:
                self.dones = {agent: True for agent in self.agents}
            
            self.agent_selection = self._agent_selector.next()
            
            self._accumulate_rewards()


## -------------------------------------------------------------------------------------------------
        def observe(self, agent_id):
            # highly recommended to reimplement this function, since the states for
            # each agent in different environments can not be standardized
            dim     = len(self._mlpro_env.get_state().get_dim_ids())
            state   = []
            for i in range(dim):
                state.append(self._mlpro_env.get_state().get_values()[i].item())
            return np.array(state, dtype=np.float32)


## -------------------------------------------------------------------------------------------------
        def reset(self):
            self.agents = self.possible_agents[:]
            self.rewards = {agent: 0 for agent in self.agents}
            self._cumulative_rewards = {agent: 0 for agent in self.agents}
            self.dones = {agent: False for agent in self.agents}
            self.infos = {agent: {} for agent in self.agents}
            self.state = {agent: None for agent in self.agents}
            self.observations = {agent: None for agent in self.agents}
            
            self._mlpro_env.reset()
            
            self._agent_selector = agent_selector(self.agents)
            self.agent_selection = self._agent_selector.next()

## -------------------------------------------------------------------------------------------------
        def render(self, mode='human'):
            try:
                if self.first_refresh:
                    self._mlpro_env.init_plot()
                    self.first_refresh = False
                else:
                    self._mlpro_env.update_plot()
                return True
            except:
                return False


## -------------------------------------------------------------------------------------------------
        def close(self):
            self._mlpro_env.__del__()
            
            
            


