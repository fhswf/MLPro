## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : multicartpole
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-06-05  0.0.0     DA       Creation
## -- 2021-06-06  1.0.0     DA       Released first version
## -- 2021-08-28  1.1.0     DA       Adjustments after changes on rl models
## -- 2021-09-11  1.1.0     MRD      Change Header information to match our new library name
## -- 2021-09-29  1.1.1     SY       Change name: WrEnvGym to WrEnvGYM2MLPro
## -- 2021-10-05  1.1.2     SY       Update following new attributes done and broken in State
## -- 2021-11-13  1.1.3     DA       Added done/broken detection
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.3 (2021-11-13)

This module provides an environment with multivariate state and action spaces based on the 
OpenAI Gym environment 'CartPole-v1'. 
"""


from mlpro.rl.models import *
from mlpro.rl.wrappers import WrEnvGYM2MLPro
import numpy as np
import gym





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MultiCartPole(Environment):
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
    def _setup_spaces(self):
        for i in range(self._num_envs):
            # Add a set of state dimensions for each sub-environment
            env_str = str(i)
            self._state_space.add_dim(Dimension(i*4, 'E-' + env_str + ' CPos', 'Env-' + env_str + ' Cart Position', '', 'm', 'm',[-4.8, 4.8]))
            self._state_space.add_dim(Dimension(i*4+1, 'E-' + env_str + ' CVel', 'Env-' + env_str + ' Cart Velocity', '', 'm/sec', '\frac{m}{sec}',[-self.C_INFINITY,self.C_INFINITY]))
            self._state_space.add_dim(Dimension(i*4+2, 'E-' + env_str + ' PAng', 'Env-' + env_str + ' Pole Angle', '', 'rad', 'rad',[-0.418, 0.418]))
            self._state_space.add_dim(Dimension(i*4+3, 'E-' + env_str + ' PAVel', 'Env-' + env_str + ' Pole Angular Velocity', '', 'rad/sec', '\frac{rad}{sec}',[-self.C_INFINITY,self.C_INFINITY]))
            
            # Add an action dimension for each sub-environment
            self._action_space.add_dim(Dimension(i, 'E-' + env_str + ' Push', 'Env-' + env_str + ' Push Cart Left/Right', '', '', '', [0,1]))

    
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
    def reset(self, p_seed=None):
        seed = p_seed

        for env in self._envs: 
            env.reset(seed)
            if seed is not None:
                seed += 1

        self._state = self.collect_substates()
  

## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_action: Action) -> None:

        self._state.set_done(True)

        for agent_id in p_action.get_agent_ids():
            action_elem = p_action.get_elem(agent_id)

            for action_id in action_elem.get_dim_ids():
                env             = self._envs[action_id]
                action_elem_env = ActionElement(env.get_action_space())
                action_elem_env.set_value(action_id, action_elem.get_value(action_id))
                action_env      = Action()
                action_env.add_elem(agent_id, action_elem_env)
                env._simulate_reaction(action_env)
                done_flag       = self.get_done() and env.get_done()
                self._state.set_done(done_flag)

        self._state = self.collect_substates()


## -------------------------------------------------------------------------------------------------
    def _evaluate_state(self) -> None: 
        for env in self._envs: env._evaluate_state()


## -------------------------------------------------------------------------------------------------
    def compute_reward(self) -> Reward:
        reward = Reward(self._reward_type)

        if self._reward_type == Reward.C_TYPE_OVERALL:
            r_overall = 0
            for env in self._envs: 
                r_overall = r_overall + env.compute_reward().get_overall_reward()
            r_overall = r_overall / len(self._envs)
            reward.set_overall_reward(r_overall)

        else:
           for agent_id in self.last_action.get_agent_ids():
                agent_action_elem = self.last_action.get_elem(agent_id)
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
    def init_plot(self, p_figure=None):
        for env in self._envs: env.init_plot(p_figure=None)


## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        for env in self._envs: env.update_plot()
