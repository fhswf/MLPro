## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.pool.envs
## -- Module  : multicartpole.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-06-05  0.0.0     DA       Creation
## -- 2021-06-06  1.0.0     DA       Released first version
## -- 2021-08-28  1.1.0     DA       Adjustments after changes on rl models
## -- 2021-09-11  1.1.0     MRD      Change Header information to match our new library name
## -- 2021-09-29  1.1.1     SY       Change name: WrEnvGym to WrEnvGYM2MLPro
## -- 2021-10-05  1.1.2     SY       Update following new attributes done and broken in State
## -- 2021-11-15  1.2.0     DA       Refactoring
## -- 2021-12-03  1.2.1     DA       Refactoring
## -- 2021-12-12  1.2.2     DA       Method MutliCartPole.get_cycle_limit() implemented
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.2 (2021-12-12)

This module provides an environment with multivariate state and action spaces based on the 
OpenAI Gym environment 'CartPole-v1'. 
"""


from mlpro.rl.models import *
from mlpro.wrappers.openai_gym import WrEnvGYM2MLPro
import numpy as np
import gym





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MultiCartPole (Environment):
    """
    This environment multivariate space and action spaces by duplicating the
    OpenAI Gym environment 'CartPole-v1'. The number of internal CartPole
    sub-enironments can be parameterized.
    """

    C_NAME      = 'MultiCartPole'
    C_LATENCY   = timedelta(0,1,0)
    C_INFINITY  = np.finfo(np.float32).max      

## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_num_envs=2,                          # Number of internal sub-environments
                 p_reward_type=Reward.C_TYPE_OVERALL,   # Reward type to be computed
                 p_logging=Log.C_LOG_ALL):              # Log level (see constants of class Log)

        self._envs           = []
        self._num_envs       = p_num_envs
        self._reward_type    = p_reward_type
        super().__init__(p_mode=Mode.C_MODE_SIM, p_logging=p_logging)
        self._state_space, self._action_space = self._setup_spaces()

        for i in range(self._num_envs): 
            state_space_env  = self._state_space.spawn([i*4, i*4+1, i*4+2, i*4+3])
            action_space_env = self._action_space.spawn([i])
            env              = WrEnvGYM2MLPro(gym.make('CartPole-v1'), state_space_env, action_space_env, p_logging=p_logging)
            env.C_NAME = env.C_NAME + ' (' + str(i) + ')'
            self._envs.append(env)
        
        self.reset()


## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging):
        super().switch_logging(p_logging)
        for env in self._envs: env.switch_logging(p_logging)


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        return None, None


## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self):
        state_space = ESpace()
        action_space = ESpace()

        for i in range(self._num_envs):
            # Add a set of state dimensions for each sub-environment
            env_str = str(i)
            state_space.add_dim(Dimension( p_id=i*4, p_name_short='E-' + env_str + ' CPos', p_name_long='Env-' + env_str + ' Cart Position', p_unit='m', p_boundaries=[-4.8, 4.8]))
            state_space.add_dim(Dimension( p_id=i*4+1, p_name_short='E-' + env_str + ' CVel', p_name_long='Env-' + env_str + ' Cart Velocity', p_unit='m/sec', p_unit_latex='\frac{m}{sec}', p_boundaries=[-self.C_INFINITY,self.C_INFINITY]))
            state_space.add_dim(Dimension( p_id=i*4+2, p_name_short='E-' + env_str + ' PAng', p_name_long='Env-' + env_str + ' Pole Angle', p_unit='rad', p_boundaries=[-0.418, 0.418]))
            state_space.add_dim(Dimension( p_id=i*4+3, p_name_short='E-' + env_str + ' PAVel', p_name_long='Env-' + env_str + ' Pole Angular Velocity', p_unit='rad/sec', p_unit_latex='\frac{rad}{sec}',p_boundaries=[-self.C_INFINITY,self.C_INFINITY]))
            
            # Add an action dimension for each sub-environment
            action_space.add_dim(Dimension( p_id=i, p_name_short='E-' + env_str + ' Push', p_name_long='Env-' + env_str + ' Push Cart Left/Right', p_boundaries=[0,1]))

        return state_space, action_space

    
## -------------------------------------------------------------------------------------------------
    def collect_substates(self) -> State:
        state = State(self._state_space)

        done    = True
        broken  = False

        for env_id, env in enumerate(self._envs):
            sub_state_val = env.get_state().get_values()
            sub_state_dim = sub_state_val.shape[0]
            for d in range(sub_state_dim):
                state.set_value(env_id*sub_state_dim + d, sub_state_val[d])

            done = done and env.get_state().get_done()
            broken = broken or env.get_state().get_broken()

        state.set_done(done)
        state.set_broken(broken)

        return state


## -------------------------------------------------------------------------------------------------
    def get_cycle_limit(self):
        return self._envs[0].get_cycle_limit()


## -------------------------------------------------------------------------------------------------
    def reset(self, p_seed=None):
        seed = p_seed

        for env in self._envs: 
            env.reset(seed)
            if seed is not None:
                seed += 1

        self._set_state( self.collect_substates() )
  

## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state:State, p_action:Action) -> State:
        done = True

        for agent_id in p_action.get_agent_ids():
            action_elem = p_action.get_elem(agent_id)

            for action_id in action_elem.get_dim_ids():
                env             = self._envs[action_id]
                action_elem_env = ActionElement(env.get_action_space())
                action_elem_env.set_value(action_id, action_elem.get_value(action_id))
                action_env      = Action()
                action_env.add_elem(agent_id, action_elem_env)
                env._set_state(env.simulate_reaction(None, action_env))
                done            = done and env.get_done()

        new_state = self.collect_substates()
        new_state.set_done(done)
        return new_state


## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old:State, p_state_new:State) -> Reward:
        reward = Reward(self._reward_type)

        if self._reward_type == Reward.C_TYPE_OVERALL:
            r_overall = 0
            for env in self._envs: 
                r_overall = r_overall + env.compute_reward().get_overall_reward()
            r_overall = r_overall / len(self._envs)
            reward.set_overall_reward(r_overall)

        else:
           for agent_id in self._last_action.get_agent_ids():
                agent_action_elem = self._last_action.get_elem(agent_id)
                agent_action_ids  = agent_action_elem.get_dim_ids()
                r_agent = 0
                for action_id in agent_action_ids:
                    r_action = self._envs[action_id].compute_reward().get_overall_reward()
                    if self._reward_type == Reward.C_TYPE_EVERY_ACTION:
                        reward.add_action_reward(agent_id, action_id, r_action)
                    elif self._reward_type == Reward.C_TYPE_EVERY_AGENT:
                        r_agent = r_agent + r_action
                
                if self._reward_type == Reward.C_TYPE_EVERY_AGENT:
                    r_agent = r_agent / len(agent_action_ids)
                    reward.add_agent_reward(agent_id, r_agent)

        return reward


## -------------------------------------------------------------------------------------------------
    def compute_done(self, p_state:State) -> bool:
        return self.get_done()


## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state:State) -> bool:
        return self.get_broken()


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        for env in self._envs: env.init_plot(p_figure=None)


## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        for env in self._envs: env.update_plot()
