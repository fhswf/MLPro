## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers
## -- Module  : pettingzoo.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-08-27  0.0.0     SY       Creation
## -- 2021-09-23  1.0.0     SY       Release of first version
## -- 2021-09-28  1.0.1     SY       WrEnvPZoo: implementation of method get_cycle_limits()
## -- 2021-09-29  1.1.0     SY       Change name:WrEnvPZoo to WrEnvPZOO2MLPro
## -- 2021-10-02  1.2.0     SY       New classes: WrEnvMLPro2PZoo, update _recognize_space() in WrEnvGYM2MLPro
## -- 2021-10-05  1.2.1     SY       Update following new attributes done and broken in State
## -- 2021-10-06  1.2.2     DA       Minor fixes
## -- 2021-10-07  1.2.3     SY       Update WrEnvMLPro2PZoo() 
## -- 2021-11-03  1.2.4     SY       Remove reset() on WrEnvPZOO2MLPro and WrEnvMLPro2PZoo to avoid double reset
## -- 2021-11-13  1.2.5     DA       Minor adjustments
## -- 2021-11-16  1.2.6     DA       Refactoring
## -- 2021-11-16  1.2.7     SY       Refactoring
## -- 2021-12-09  1.2.8     SY       Update process action procedure in WrEnvMLPro2PZoo()
## -- 2021-12-11  1.2.9     SY       Update WrEnvPZOO2MLPro() in setting up done flag
## -- 2021-12-21  1.3.0     DA       - Replaced 'done' by 'success' on mlpro functionality
## --                                - Optimized 'done' detection in both classes
## -- 2021-12-23  1.3.1     MRD      Remove adding self._num_cycle on simulate_reaction() due to 
## --                                EnvBase.process_actions() is already adding self._num_cycle
## -- 2022-01-20  1.3.2     SY       - Update PettingZoo2MLPro's reward type to C_TYPE_EVERY_AGENT 
## --                                - Update Wrapper MLPro2PettingZoo - Method step()
## -- 2022-01-21  1.3.3     SY       Class WrEnvPZOO2MLPro: 
## --                                - replace variable _reward to _last_reward 
## --                                Class WrEnvMLPro2PZoo:  
## --                                - refactored done detection 
## --                                - removed artifacts of cycle counting
## -- 2022-02-27  1.3.4     SY       Refactoring due to auto generated ID in class Dimension
## -- 2022-03-21  1.3.5     SY       Refactoring due to PettingZoo version 1.17.0
## -- 2022-05-20  1.3.6     SY       Refactoring: Action space boundaries in WrEnvPZOO2MLPro
## -- 2022-05-30  1.3.7     SY       Replace function env.seed(seed) to env.reset(seed=seed)
## -- 2022-07-20  1.3.8     SY       Update due to the latest introduction of Gym 0.25
## -- 2022-08-15  1.4.0     DA       Introduction of root class Wrapper
## -- 2022-09-26  1.5.0     SY       Update following PettingZoo version 1.21.0 and Gym 0.26
## -- 2022-10-06  2.0.0     SY       Major update following PettingZoo version 1.22.0
## -- 2022-11-01  2.0.1     DA       Refactoring
## -- 2022-11-09  2.0.2     DA       Refactoring
## -- 2022-11-29  2.0.3     DA       Refactoring
## -- 2022-12-21  2.0.4     SY       Update _recognize_space due to howto_rl_wp_003
## -- 2023-02-21  2.1.0     DA       Class WrEnvPZOO2MLPro: specific implementations for load(), _save()
## -- 2023-03-26  2.2.0     DA       Class WrEnvPZOO2MLPro: refactoring of persistence
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.2.0 (2023-03-26)

This module provides wrapper classes for PettingZoo multi-agent environments.

See also: https://pypi.org/project/PettingZoo/

"""


import gymnasium
from pydoc import locate
import numpy as np
from mlpro.wrappers.models import Wrapper
from mlpro.rl import *
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrEnvPZOO2MLPro(Wrapper, Environment):
    """
    This class is a ready to use wrapper class for Petting Zoo environments. 
    Objects of this type can be treated as an environment object. Encapsulated 
    petting zoo environment must be compatible to class pettingzoo.env.

    Parameters
    ----------
    p_pzoo_env
        Petting Zoo environment object
    p_state_space : MSpace
        Optional external state space object that meets the state space of the gym environment
    p_action_space : MSpace 
        Optional external action space object that meets the action space of the gym environment
    p_visualize : bool
        Boolean switch for env/agent visualisation. Default = True.
    p_logging
        Log level (see constants of class Log). Default = Log.C_LOG_ALL.
    """

    C_TYPE              = 'Wrapper PettingZoo2MLPro'
    C_WRAPPED_PACKAGE   = 'pettingzoo'
    C_MINIMUM_VERSION   = '1.20.0'
    C_PLOT_ACTIVE: bool = True

    C_SUPPORTED_MODULES = [ 'pettingzoo.classic',
                            'pettingzoo.butterfly',
                            'pettingzoo.atari',
                            'pettingzoo.mpe',
                            'pettingzoo.sisl' ]

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_zoo_env, 
                  p_state_space:MSpace=None, 
                  p_action_space:MSpace=None, 
                  p_visualize:bool=True,
                  p_logging=Log.C_LOG_ALL):

        self._zoo_env          = p_zoo_env
        self._zoo_env_cls      = None
        self._zoo_env_cls_name = self._zoo_env.metadata['name']

        for mod in self.C_SUPPORTED_MODULES:
            try:
                self._zoo_env_cls = locate( mod + '.' + self._zoo_env_cls_name )
                break
            except:
                pass

        if self._zoo_env_cls is None:
            raise ParamError('PettingZoo environment class "' + self._zoo_env_cls_name + '" not supported!')

        self.C_NAME       = 'Env "' + self._zoo_env.metadata['name'] + '"'

        Environment.__init__(self, p_mode=Environment.C_MODE_SIM, p_visualize=p_visualize, p_logging=p_logging)
        Wrapper.__init__(self, p_logging=p_logging)
        
        if p_state_space is not None: 
            self._state_space = p_state_space
        else:
            self._state_space = self._recognize_space(self._zoo_env.observation_spaces, "observation")
        
        if p_action_space is not None: 
            self._action_space = p_action_space
        else:
            self._action_space = self._recognize_space(self._zoo_env.action_spaces, "action")


## -------------------------------------------------------------------------------------------------
    def __del__(self):
        try:
            self._zoo_env.close()
        except:
            pass
        
        self.log(self.C_LOG_TYPE_I, 'Closed')


## -------------------------------------------------------------------------------------------------
    def _reduce_state(self, p_state:dict, p_path:str, p_os_sep:str, p_filename_stub:str):
        """
        The embedded PettingZoo env itself can't be pickled due to it's dependencies on Pygame. That's 
        why the current env instance needs to be removed before pickling the object. 

        See also: https://stackoverflow.com/questions/52336196/how-to-save-object-using-pygame-surfaces-to-file-using-pickle
        """

        p_state['_zoo_env'] = None


## -------------------------------------------------------------------------------------------------
    def _complete_state(self, p_path:str, p_os_sep:str, p_filename_stub:str):

        if self.get_visualization():
            render_mode = 'human'
        else:
            render_mode = 'ansi'

        for mod in WrEnvPZOO2MLPro.C_SUPPORTED_MODULES:
            try:
                self._zoo_env_cls = locate( mod + '.' + self._zoo_env_cls_name )
                break
            except:
                pass

        self._zoo_env = self._zoo_env_cls.env(render_mode=render_mode)


## -------------------------------------------------------------------------------------------------
    def _recognize_space(self, p_zoo_space, dict_name) -> ESpace:
        space = ESpace()
        d_idx = 0
        
        if dict_name == "observation":
            space.add_dim(Dimension(p_name_short='0', p_base_set='DO'))
        elif dict_name == "action":
            for k in p_zoo_space:
                if isinstance(p_zoo_space[k], gymnasium.spaces.Discrete):
                    space.add_dim(Dimension(p_name_short=k, p_base_set=Dimension.C_BASE_SET_Z,
                                            p_boundaries=[0, int(p_zoo_space[k].n-1)]))
                else:
                    try:
                        shape_dim = len(p_zoo_space[k].shape)
                        for i in range(shape_dim):
                            for d in range(p_zoo_space[k].shape[i]):
                                space.add_dim(Dimension(p_name_short=str(d_idx), p_base_set=Dimension.C_BASE_SET_R,
                                                        p_boundaries=[p_zoo_space[k].low[d], p_zoo_space[k].high[d]]))
                                d_idx += 1
                    except:
                        space.add_dim(Dimension(p_name_short=k, p_base_set='DO'))
                
        return space


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        return None, None


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None):

        # 1 Reset Zoo environment and determine initial state
        try:
            self._zoo_env.reset(seed=p_seed)
        except:
            self._zoo_env.seed(p_seed)
            self._zoo_env.reset()
        observation, _, _, _, _ = self._zoo_env.last()
        obs     = DataObject(observation)
        
        # 2 Create state object from Zoo observation
        state   = State(self._state_space)
        if isinstance(observation, dict):
            state.set_values(obs.get_data()['observation'])
        else:
            state.set_values(obs.get_data())
        self._set_state(state)


## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state:State, p_action:Action) -> State:

        new_state = State(self._state_space)

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
            observation, reward_zoo, termination, truncation, info = self._zoo_env.last()

            obs     = DataObject(observation)
            
            if termination or truncation:
                self._zoo_env.step(None)
                new_state.set_terminal(True)

            else:
                try:
                    self._zoo_env.step(action_zoo)
                except:
                    self._zoo_env.step(np.atleast_1d(action_zoo))
            agent_num += 1

        # 3 Create state object from Zoo observation
            if isinstance(observation, dict):
                new_state.set_values(obs.get_data()['observation'])
            else:
                new_state.set_values(obs.get_data())

        # 4 Create and store reward object
        self._last_reward = Reward(Reward.C_TYPE_EVERY_AGENT)
        for key in self._zoo_env.rewards.keys():
            self._last_reward.add_agent_reward(key, self._zoo_env.rewards.get(key))

        return new_state


## -------------------------------------------------------------------------------------------------
    def compute_reward(self, p_state_old:State=None, p_state_new:State=None) -> Reward:
        if ( p_state_old is not None ) or ( p_state_new is not None ):
            raise NotImplementedError

        return self._last_reward


## -------------------------------------------------------------------------------------------------
    def compute_success(self, p_state:State) -> bool:
        return self.get_success()


## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state:State) -> bool:
        return self.get_broken()


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure: Figure = None, p_plot_settings: list = ..., p_plot_depth: int = 0, p_detail_level: int = 0, p_step_rate: int = 0, **p_kwargs):
        if self._visualize: self._zoo_env.render()


## -------------------------------------------------------------------------------------------------
    def update_plot(self, **p_kwargs):
        if self._visualize: self._zoo_env.render()


## -------------------------------------------------------------------------------------------------
    def get_cycle_limit(self):
        try:
            return self._zoo_env.env.env.max_cycles
        except:
            return self.C_CYCLE_LIMIT





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrEnvMLPro2PZoo(Wrapper):
    """
    This class is a ready to use wrapper class for MLPro to PettingZoo environments. 
    Objects of this type can be treated as an AECEnv object. Encapsulated 
    MLPro environment must be compatible to class Environment.
    To be noted, this wrapper is not capable for parallel environment yet.

    Parameters
    ----------
    p_mlpro_env : Environment    
        MLPro's Environment object
    p_num_agents : int   
        Number of Agents
    p_state_space : MSpace  
        Optional external state space object that meets the state space of the MLPro environment
    p_action_space : MSpace  
        Optional external action space object that meets the action space of the MLPro environment
    """

    C_TYPE              = 'Wrapper MLPro2PettingZoo'
    C_WRAPPED_PACKAGE   = 'pettingzoo'
    C_MINIMUM_VERSION   = '1.20.0'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_mlpro_env:Environment, p_num_agents, p_state_space:MSpace=None, p_action_space:MSpace=None, p_logging=Log.C_LOG_ALL):       
        self.C_NAME = 'Env "' + p_mlpro_env.C_NAME + '"'
        Wrapper.__init__(self, p_logging=p_logging)
        self.pzoo_env   = self.raw_env(p_mlpro_env, p_num_agents, p_state_space, p_action_space)
        if self.pzoo_env.render_mode == 'ansi':
            self.pzoo_env   = wrappers.CaptureStdoutWrapper(self.pzoo_env)
        self.pzoo_env   = wrappers.OrderEnforcingWrapper(self.pzoo_env)


## -------------------------------------------------------------------------------------------------
    class raw_env(AECEnv):
        metadata = {'render_modes': ['human', 'ansi'], "name": "pzoo_custom"}

## -------------------------------------------------------------------------------------------------
        def __init__(self, p_mlpro_env, p_num_agents, p_state_space:MSpace=None, p_action_space:MSpace=None, p_render_mode='human'):
            self._mlpro_env             = p_mlpro_env
            self.possible_agents        = [str(r) for r in range(p_num_agents)]
            self.agent_name_mapping     = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
            self.render_mode            = p_render_mode
            
            if p_state_space is not None: 
                self.observation_spaces = p_state_space
            else:
                self.observation_spaces = self._recognize_space(self._mlpro_env.get_state_space())
            
            if p_action_space is not None: 
                self.action_spaces      = p_action_space
            else:
                self.action_spaces      = self._recognize_space(self._mlpro_env.get_action_space())
            
            self.first_refresh          = True
        

## -------------------------------------------------------------------------------------------------
        def _recognize_space(self, p_mlpro_space):
            space = None
            action_dim = p_mlpro_space.get_num_dim()
            id_dim = p_mlpro_space.get_dim_ids()[0]
            base_set = p_mlpro_space.get_dim(id_dim).get_base_set()
            if len(p_mlpro_space.get_dim(id_dim).get_boundaries()) == 1:
                space = gymnasium.spaces.Discrete(p_mlpro_space.get_dim(id_dim).get_boundaries()[0]+1)
            elif base_set == Dimension.C_BASE_SET_Z or base_set == Dimension.C_BASE_SET_N:
                low_limit = p_mlpro_space.get_dim(id_dim).get_boundaries()[0]
                up_limit = p_mlpro_space.get_dim(id_dim).get_boundaries()[1]
                num_discrete = int(up_limit-low_limit+1)
                space = gymnasium.spaces.Discrete(num_discrete)
            else:
                lows = []
                highs = []
                for dimension in range(action_dim):
                    id_dim = p_mlpro_space.get_dim_ids()[dimension]
                    lows.append(p_mlpro_space.get_dim(id_dim).get_boundaries()[0])
                    highs.append(p_mlpro_space.get_dim(id_dim).get_boundaries()[1])
    
                space = gymnasium.spaces.Box(
                    low=np.array(lows, dtype=np.float32),
                    high=np.array(highs, dtype=np.float32),
                    shape=(action_dim,),
                    dtype=np.float32
                )
                
            setup_space     = {agent: space for agent in self.possible_agents}
                
            return setup_space


## -------------------------------------------------------------------------------------------------
        def step(self, action):
            if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
                self._was_dead_step(action)
                return
            
            agent = self.agent_selection
            self._cumulative_rewards[agent] = 0
            cycle_limit = self._mlpro_env.get_cycle_limit()
            
            self.state[self.agent_selection] = action[int(self.agent_selection)]
            
            if agent == self.possible_agents[-1]:
                _action     = Action()
                idx         = self._mlpro_env.get_action_space().get_num_dim()
                if isinstance(self.observation_spaces, gymnasium.spaces.Discrete):
                    action = np.array([action])
                for i in range(idx):
                    _act_set    = Set()
                    _act_set.add_dim(Dimension('action_'+str(i)))
                    _act_elem   = Element(_act_set)
                    _ids = _act_elem.get_dim_ids()
                    _act_elem.set_value(_ids[0], self.state[self.possible_agents[i]])
                    _action.add_elem(self.possible_agents[i], _act_elem)
                
                self._mlpro_env.process_action(_action)
                for i in range(idx):
                    self.rewards[self.possible_agents[i]] = self._mlpro_env.compute_reward().get_agent_reward(self.possible_agents[i]).item()
                    if not self.rewards[self.possible_agents[i]]:
                        self.rewards[self.possible_agents[i]] = 0
            
                if self._mlpro_env.get_state().get_terminal():
                    self.truncations = {agent: True for agent in self.agents}
                    self.terminations = {agent: True for agent in self.agents}
                    
                for i in self.agents:
                    self.observations[i] = self.state[self.agents[1-int(i)]]
            
            else:
                self._clear_rewards()
                
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
        def reset(self, seed, options):
            self.agents = self.possible_agents[:]
            self.rewards = {agent: 0 for agent in self.agents}
            self._cumulative_rewards = {agent: 0 for agent in self.agents}
            self.terminations = {agent: False for agent in self.agents}
            self.truncations = {agent: False for agent in self.agents}
            self.infos = {agent: {} for agent in self.agents}
            self.state = {agent: None for agent in self.agents}
            self.observations = {agent: None for agent in self.agents}
            
            self._mlpro_env.reset(seed)
            
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