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
## -- 2021-10-05  1.3.1     SY       Update following new attributes done and broken in State
## -- 2021-10-06  1.3.2     DA       Minor fixes
## -- 2021-10-07  1.3.3     MRD      Redefine WrEnvMLPro2GYM reset(), step(), _recognize_space() function
## --                                Redefine also _recognize_space() from WrEnvGYM2MLPro
## -- 2021-10-07  1.3.4     SY       Update WrEnvMLPro2PZoo() following above changes (ver. 1.3.3)
## -- 2021-10-07  1.4.0     MRD      Implement WrPolicySB32MLPro to wrap the policy from Stable-baselines3
## -- 2021-10-08  1.4.1     DA       Correction of wrapper WREnvGYM2MLPro
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.0 (2021-10-07)

This module provides wrapper classes for reinforcement learning tasks.
"""


import gym
from gym import error, spaces, utils
import numpy as np
import torch
from stable_baselines3.common.logger import Logger
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
        
        if isinstance(p_gym_space, gym.spaces.Discrete):
            space.add_dim(Dimension(p_id=0,p_name_short='0', p_base_set=Dimension.C_BASE_SET_Z, p_boundaries=[p_gym_space.n]))
        elif isinstance(p_gym_space, gym.spaces.Box):
            shape_dim = len(p_gym_space.shape)
            for i in range(shape_dim):
                for d in range(p_gym_space.shape[i]):
                    space.add_dim(Dimension(p_id=d, p_name_short=str(d), p_base_set=Dimension.C_BASE_SET_R, p_boundaries=[p_gym_space.low[d], p_gym_space.high[d]]))

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
            observation, reward_gym, done, info = self._gym_env.step(action_gym)
        except:
            observation, reward_gym, done, info = self._gym_env.step(np.atleast_1d(action_gym))
        
        obs     = DataObject(observation)

        # 3 Create state object from Gym observation
        state   = State(self._state_space)
        state.set_values(obs.get_data())
        state.set_done(done)
        self._set_state(state)

        # 4 Create and store reward object
        self.reward = Reward(Reward.C_TYPE_OVERALL)
        self.reward.set_overall_reward(reward_gym)


## -------------------------------------------------------------------------------------------------
    def _evaluate_state(self):
        if self.get_done():
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
            observation, reward_zoo, done, info = self._zoo_env.last()
            self._state.set_done(done)
            obs     = DataObject(observation)
            
            if self.get_done():
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
        if self.get_done():
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
            self.observation_space  = self._recognize_space(self._mlpro_env.get_state_space())
        
        if p_action_space is not None: 
            self.action_space       = p_action_space
        else:
            self.action_space       = self._recognize_space(self._mlpro_env.get_action_space())
        
        self.first_refresh          = True
        self.reset()
        

## -------------------------------------------------------------------------------------------------
    def _recognize_space(self, p_mlpro_space):
        space = None
        action_dim = p_mlpro_space.get_num_dim()
        if len(p_mlpro_space.get_dim(0).get_boundaries()) == 1:
            space = gym.spaces.Discrete(p_mlpro_space.get_dim(0).get_boundaries()[0])
        else:
            lows = []
            highs = []
            for dimension in range(action_dim):
                lows.append(p_mlpro_space.get_dim(dimension).get_boundaries()[0])
                highs.append(p_mlpro_space.get_dim(dimension).get_boundaries()[1])

            space = gym.spaces.Box(
                            low=np.array(lows, dtype=np.float32), 
                            high=np.array(highs, dtype=np.float32), 
                            shape=(action_dim,), 
                            dtype=np.float32
                            )
            
        return space


## -------------------------------------------------------------------------------------------------
    def step(self, action):
        _action     = Action()
        _act_set    = Set()
        idx         = self._mlpro_env._action_space.get_num_dim()

        if isinstance(self.observation_space, gym.spaces.Discrete):
            action = np.array([action])
        
        for i in range(idx):
            _act_set.add_dim(Dimension(i,'action_'+str(i)))
        _act_elem   = Element(_act_set)
        for i in range(idx):
            _act_elem.set_value(i, action[i].item())
        _action.add_elem('0', _act_elem)
        
        self._mlpro_env.process_action(_action)
        reward          = self._mlpro_env.compute_reward()
        self._mlpro_env._evaluate_state
        
        obs = None
        if isinstance(self.observation_space, gym.spaces.Box):
            obs = np.array(self._mlpro_env.get_state().get_values(), dtype=np.float32)
        else:
            obs = np.array(self._mlpro_env.get_state().get_values())
        return obs, reward.get_overall_reward(), self._mlpro_env.get_done(), {}
    

## -------------------------------------------------------------------------------------------------
    def reset(self):
        self._mlpro_env.reset()
        obs = None
        if isinstance(self.observation_space, gym.spaces.Box):
            obs = np.array(self._mlpro_env.get_state().get_values(), dtype=np.float32)
        else:
            obs = np.array(self._mlpro_env.get_state().get_values())
        return obs
    

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
                self.observation_spaces = self._recognize_space(self._mlpro_env.get_state_space())
            
            if p_action_space is not None: 
                self.action_spaces      = p_action_space
            else:
                self.action_spaces      = self._recognize_space(self._mlpro_env.get_action_space())
            
            self.first_refresh          = True
            self.reset()
        

## -------------------------------------------------------------------------------------------------
        def _recognize_space(self, p_mlpro_space):
            space = None
            action_dim = p_mlpro_space.get_num_dim()
            if len(p_mlpro_space.get_dim(0).get_boundaries()) == 1:
                space = gym.spaces.Discrete(p_mlpro_space.get_dim(0).get_boundaries()[0])
            else:
                lows = []
                highs = []
                for dimension in range(action_dim):
                    lows.append(p_mlpro_space.get_dim(dimension).get_boundaries()[0])
                    highs.append(p_mlpro_space.get_dim(dimension).get_boundaries()[1])
    
                space = gym.spaces.Box(
                                low=np.array(lows, dtype=np.float32), 
                                high=np.array(highs, dtype=np.float32), 
                                shape=(action_dim,), 
                                dtype=np.float32
                                )
                
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
                idx         = self._mlpro_env.get_action_space().get_num_dim()
                if isinstance(self.observation_space, gym.spaces.Discrete):
                    action = np.array([action])
                for i in range(idx):
                    _act_set.add_dim(Dimension(i,'action_'+str(i)))
                _act_elem   = Element(_act_set)
                for i in range(idx):
                    _act_elem.set_value(i, self.action_set[i])
                _action.add_elem('0', _act_elem)
                
                self._mlpro_env.process_action(_action)
                self._mlpro_env._evaluate_state()
                
            self.rewards[agent] = self._mlpro_env.compute_reward().get_agent_reward(agent)
            
            if self._mlpro_env.get_done():
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
            
## -------------------------------------------------------------------------------------------------
class WrPolicySB32MLPro(Policy):
    """
    This class provides a policy wrapper from Standard Baselines 3 (SB3).
    Especially On-Policy Algorithm
    """
    C_TYPE        = 'SB3 Policy'

    def __init__(self, p_sb3_policy, p_state_space, p_action_space, p_buffer_size, p_ada=True, p_logging=True):
        """
        Args:
            p_sb3_policy : SB3 Policy
            p_state_space : Environment State Space
            p_action_space : Environment Action Space
            p_buffer_size : Buffer Size
            p_ada (bool, optional): Adaptability. Defaults to True.
            p_logging (bool, optional): Logging. Defaults to True.
        """

        class EmptyLogger(Logger):
            """
            Dummy class for SB3 Empty Logger. This is due to that SB3 has its own logger class.
            Since we wont be using SB3 logger, we need Empty Logger to run the SB3 train, 
            otherwise it wont work.
            """
            def __init__():
                super().__init__()
            def record(self, key=None, value=None, exclude=None):
                pass

        super().__init__(p_state_space, p_action_space, p_buffer_size=p_buffer_size, p_ada=p_ada, p_logging=p_logging)
        
        self.sb3 = p_sb3_policy
        self.last_buffer_element = None
            

        # Variable preparation for SB3
        action_space = None
        state_space = None

        # Check if action is Discrete or Box
        action_dim = self.get_action_space().get_num_dim()
        if len(self.get_action_space().get_dim(0).get_boundaries()) == 1:
            action_space = gym.spaces.Discrete(self.get_action_space().get_dim(0).get_boundaries()[0])
        else:
            lows = []
            highs = []
            for dimension in range(action_dim):
                lows.append(self.get_action_space().get_dim(dimension).get_boundaries()[0])
                highs.append(self.get_action_space().get_dim(dimension).get_boundaries()[1])

            action_space = gym.spaces.Box(
                            low=np.array(lows, dtype=np.float32), 
                            high=np.array(highs, dtype=np.float32), 
                            shape=(action_dim,), 
                            dtype=np.float32
                            )

        # Check if state is Discrete or Box
        state_dim = self.get_state_space().get_num_dim()
        if len(self.get_state_space().get_dim(0).get_boundaries()) == 1:
            state_space = gym.spaces.Discrete(self.get_state_space().get_dim(0).get_boundaries()[0])
        else:
            lows = []
            highs = []
            for dimension in range(state_dim):
                lows.append(self.get_state_space().get_dim(dimension).get_boundaries()[0])
                highs.append(self.get_state_space().get_dim(dimension).get_boundaries()[1])

            state_space = gym.spaces.Box(
                            low=np.array(lows, dtype=np.float32), 
                            high=np.array(highs, dtype=np.float32), 
                            shape=(state_dim,), 
                            dtype=np.float32
                            )

        # Setup SB3 Model
        self.sb3.observation_space = state_space
        self.sb3.action_space = action_space
        self.sb3.n_steps = p_buffer_size
        self.sb3.n_envs = 1

        self.sb3._setup_model()
        self.sb3.set_logger(EmptyLogger)

        self._buffer = self.sb3.rollout_buffer

    def compute_action(self, p_state: State) -> Action:
        obs = p_state.get_values()
        
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs).reshape(1,obs.size).to(self.sb3.device)
        
        with torch.no_grad():
            actions, values, log_probs = self.sb3.policy.forward(obs)
        
        # Add to additional_buffer_element
        self.additional_buffer_element = dict(value=values, action_log=log_probs)

        action = actions.cpu().numpy().flatten()
        action = Action(self._id, self._action_space, action)
        return action

    def _adapt(self, *p_args) -> bool:
        # Add to buffer
        self.add_buffer(p_args[0])

        # Adapt only when Buffer is full
        if not self.sb3.rollout_buffer.full:
            self.log(self.C_LOG_TYPE_I, 'Buffer is not full yet, keep collecting data!')
            return False

        last_obs = torch.Tensor([self.last_buffer_element.get_data()["state_new"].get_values()]).to(self.sb3.device)
        last_done = self.last_buffer_element.get_data()["state_new"].get_done()

        # Get the next value from the last observation
        with torch.no_grad():
            _, last_values, _ = self.sb3.policy.forward(last_obs)

        # Compute Return and Advantage
        self.sb3.rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=last_done)

        # Train
        self.sb3.train()

        # Clear Buffer After Update
        self.sb3.rollout_buffer.reset()

        return True
    
    def clear_buffer(self):
        self.sb3.rollout_buffer.reset()
    
    def add_buffer(self, p_buffer_element: SARSElement):
        """
        Redefine add_buffer function. Instead of adding to MLPro SARBuffer, we are using
        internal buffer from SB3.
        """
        self.last_buffer_element = self._add_additional_buffer(p_buffer_element)
        datas = self.last_buffer_element.get_data()
        self.sb3.rollout_buffer.add(
                            datas["state"].get_values(),
                            datas["action"].get_sorted_values(),
                            datas["reward"].get_overall_reward(),
                            datas["state"].get_done(),
                            datas["value"],
                            datas["action_log"])

        self._buffer = self.sb3.rollout_buffer

    def _add_additional_buffer(self, p_buffer_element: SARSElement):
        p_buffer_element.add_value_element(self.additional_buffer_element)
        return p_buffer_element
