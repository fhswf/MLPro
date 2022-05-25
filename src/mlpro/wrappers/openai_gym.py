## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers
## -- Module  : openai_gym.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-08-27  0.0.0     SY       Creation
## -- 2021-08-27  1.0.0     SY       Release of first version
## -- 2021-09-11  1.0.0     MRD      Change Header information to match our new library name
## -- 2021-09-23  1.1.0     SY       Update WrEnvGym to solve big data issues
## -- 2021-09-24  1.1.1     MRD      Change the gym wrapper _recognize_space() function to seperate
## --                                between discrete space and continuous space
## -- 2021-09-28  1.1.2     SY       WrEnvGym: implementation of method get_cycle_limits()
## -- 2021-09-29  1.1.3     SY       Change name: WrEnvGym to WrEnvGYM2MLPro
## -- 2021-09-30  1.2.0     SY       New classes: WrEnvMLPro2GYM
## -- 2021-10-05  1.2.1     SY       Update following new attributes done and broken in State
## -- 2021-10-06  1.2.2     DA       Minor fixes
## -- 2021-10-07  1.2.3     MRD      Redefine WrEnvMLPro2GYM reset(), step(), _recognize_space() function
## --                                Redefine also _recognize_space() from WrEnvGYM2MLPro
## -- 2021-10-08  1.2.4     DA       Correction of wrapper WREnvGYM2MLPro
## -- 2021-10-27  1.2.5     MRD      Remove reset() on WREnvGYM2MLPro and WrEnvMLPro2GYM init() function
## --                                to prevent double reset, due to it will be reset later on Training Class
## --                                Mismatch datatype last_done on WrPolicySB32MLPro
## -- 2021-11-13  1.2.6     DA       Minor adjustments
## -- 2021-11-16  1.2.7     DA       Refactoring
## -- 2021-11-16  1.2.8     SY       Refactoring
## -- 2021-12-03  1.2.9     DA       Refactoring
## -- 2021-12-21  1.3.0     DA       - Replaced 'done' by 'success' on mlpro functionality
## --                                - Optimized 'done' detection in both classed
## -- 2021-12-23  1.3.1     MRD      Remove adding self._num_cycle on simulate_reaction() due to 
## --                                EnvBase.process_actions() is already adding self._num_cycle
## -- 2022-01-21  1.3.2     DA/MRD   Class WrEnvMLPro2GYM: 
## --                                - refactored done detection 
## --                                - removed artifacts of cycle counting
## -- 2022-01-28  1.3.3     DA       Class WrEnvMLPro2GYM: stabilized destructor
## -- 2022-02-27  1.3.4     SY       Refactoring due to auto generated ID in class Dimension
## -- 2022-03-21  1.3.5     MRD      Added new parameter to the WrEnvMLPro2GYM.reset()
## -- 2022-05-19  1.3.6     SY       Gym 0.23: Replace function env.seed(seed) to env.reset(seed=seed)
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.6 (2022-05-19)

This module provides wrapper classes for reinforcement learning tasks.
"""

import gym
from mlpro.rl.models import *


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrEnvGYM2MLPro(Environment):
    """
    This class is a ready to use wrapper class for OpenAI Gym environments. 
    Objects of this type can be treated as an environment object. Encapsulated 
    gym environment must be compatible to class gym.Env.
    """

    C_TYPE = 'OpenAI Gym Env'

    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_gym_env,  # Gym environment object
                 p_state_space: MSpace = None,  # Optional external state space object that meets the
                 # state space of the gym environment
                 p_action_space: MSpace = None,  # ptional external action space object that meets the
                 # state space of the gym environment
                 p_logging=Log.C_LOG_ALL):  # Log level (see constants of class Log)

        self._gym_env = p_gym_env
        self.C_NAME = 'Env "' + self._gym_env.spec.id + '"'

        super().__init__(p_mode=Environment.C_MODE_SIM, p_latency=None, p_logging=p_logging)

        if p_state_space is not None:
            self._state_space = p_state_space
        else:
            self._state_space = self.recognize_space(self._gym_env.observation_space)

        if p_action_space is not None:
            self._action_space = p_action_space
        else:
            self._action_space = self.recognize_space(self._gym_env.action_space)

    ## -------------------------------------------------------------------------------------------------
    def __del__(self):
        try:
            self._gym_env.close()
            self.log(self.C_LOG_TYPE_I, 'Closed')
        except:
            pass

    ## -------------------------------------------------------------------------------------------------
    @staticmethod
    def recognize_space(p_gym_space) -> ESpace:
        space = ESpace()

        if isinstance(p_gym_space, gym.spaces.Discrete):
            space.add_dim(
                Dimension(p_name_short='0', p_base_set=Dimension.C_BASE_SET_Z, p_boundaries=[p_gym_space.n]))
        elif isinstance(p_gym_space, gym.spaces.Box):
            shape_dim = len(p_gym_space.shape)
            for i in range(shape_dim):
                for d in range(p_gym_space.shape[i]):
                    space.add_dim(Dimension(p_name_short=str(d), p_base_set=Dimension.C_BASE_SET_R,
                                            p_boundaries=[p_gym_space.low[d], p_gym_space.high[d]]))

        return space

    ## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        return None, None

    ## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None):

        # 1 Reset Gym environment and determine initial state
        try:
            observation = self._gym_env.reset(seed=p_seed)
        except:
           self._gym_env.seed(p_seed)
           observation = self._gym_env.reset() 
        obs = DataObject(observation)

        # 2 Create state object from Gym observation
        state = State(self._state_space)
        state.set_values(obs.get_data())
        self._set_state(state)

    ## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state: State, p_action: Action) -> State:

        # 1 Convert action to Gym syntax
        action_sorted = p_action.get_sorted_values()
        dtype = self._gym_env.action_space.dtype

        if (dtype == np.int32) or (dtype == np.int64):
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

        obs = DataObject(observation)

        # 3 Create state object from Gym observation
        state = State(self._state_space, p_terminal=done)
        state.set_values(obs.get_data())

        # 4 Create reward object
        self._last_reward = Reward(Reward.C_TYPE_OVERALL)
        self._last_reward.set_overall_reward(reward_gym)

        # 5 Return next state
        return state

    ## -------------------------------------------------------------------------------------------------
    def compute_reward(self, p_state_old: State = None, p_state_new: State = None) -> Reward:
        if (p_state_old is not None) or (p_state_new is not None):
            raise NotImplementedError

        return self._last_reward

    ## -------------------------------------------------------------------------------------------------
    def compute_success(self, p_state: State) -> bool:
        return self.get_success()

    ## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state: State) -> bool:
        return self.get_broken()

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
class WrEnvMLPro2GYM(gym.Env):
    """
    This class is a ready to use wrapper class for MLPro to OpenAI Gym environments. 
    Objects of this type can be treated as an gym.Env object. Encapsulated 
    MLPro environment must be compatible to class Environment.
    """

    C_TYPE = 'MLPro to Gym Env'
    metadata = {'render.modes': ['human']}

    ## -------------------------------------------------------------------------------------------------
    def __init__(self, p_mlpro_env, p_state_space: MSpace = None, p_action_space: MSpace = None):
        """
        Parameters:
            p_mlpro_env     MLPro's Environment object
            p_state_space   Optional external state space object that meets the
                            state space of the MLPro environment
            p_action_space  Optional external action space object that meets the
                            state space of the MLPro environment
        """

        self._mlpro_env = p_mlpro_env

        if p_state_space is not None:
            self.observation_space = p_state_space
        else:
            self.observation_space = self.recognize_space(self._mlpro_env.get_state_space())

        if p_action_space is not None:
            self.action_space = p_action_space
        else:
            self.action_space = self.recognize_space(self._mlpro_env.get_action_space())

        self.first_refresh = True

    ## -------------------------------------------------------------------------------------------------
    @staticmethod
    def recognize_space(p_mlpro_space):
        space = None
        action_dim = p_mlpro_space.get_num_dim()
        id_dim = p_mlpro_space.get_dim_ids()[0]
        if len(p_mlpro_space.get_dim(id_dim).get_boundaries()) == 1:
            space = gym.spaces.Discrete(p_mlpro_space.get_dim(id_dim).get_boundaries()[0])
        else:
            lows = []
            highs = []
            for dimension in range(action_dim):
                id_dim = p_mlpro_space.get_dim_ids()[dimension]
                lows.append(p_mlpro_space.get_dim(id_dim).get_boundaries()[0])
                highs.append(p_mlpro_space.get_dim(id_dim).get_boundaries()[1])

            space = gym.spaces.Box(
                low=np.array(lows, dtype=np.float32),
                high=np.array(highs, dtype=np.float32),
                shape=(action_dim,),
                dtype=np.float32
            )

        return space

    ## -------------------------------------------------------------------------------------------------
    def step(self, action):
        _action = Action()
        _act_set = Set()
        idx = self._mlpro_env._action_space.get_num_dim()

        if isinstance(self.observation_space, gym.spaces.Discrete):
            action = np.array([action])

        for i in range(idx):
            _act_set.add_dim(Dimension('action_' + str(i)))
        _act_elem = Element(_act_set)
        for i in range(idx):
            _ids = _act_elem.get_dim_ids()
            _act_elem.set_value(_ids[i], action[i].item())
        _action.add_elem('0', _act_elem)

        self._mlpro_env.process_action(_action)
        reward = self._mlpro_env.compute_reward()

        obs = None
        if isinstance(self.observation_space, gym.spaces.Box):
            obs = np.array(self._mlpro_env.get_state().get_values(), dtype=np.float32)
        else:
            obs = np.array(self._mlpro_env.get_state().get_values())

        state = self._mlpro_env.get_state()
        done = state.get_terminal()

        info = {}
        info["TimeLimit.truncated"] = state.get_timeout()

        return obs, reward.get_overall_reward(), done, info

    ## -------------------------------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        self._mlpro_env.reset(seed)
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