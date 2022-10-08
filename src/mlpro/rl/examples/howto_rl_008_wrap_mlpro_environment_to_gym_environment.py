## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl
## -- Module  : howto_rl_008_wrap_mlpro_environment_to_gym_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-30  0.0.0     SY       Creation
## -- 2021-09-30  1.0.0     SY       Released first version
## -- 2021-10-04  1.0.1     DA       Minor fixes
## -- 2021-12-22  1.0.2     DA       Cleaned up a bit
## -- 2022-03-21  1.0.3     MRD      Use Gym Env Checker
## -- 2022-05-30  1.0.4     DA       Little refactoring
## -- 2022-07-28  1.0.5     SY       Update due to the latest introduction of Gym 0.25
## -- 2022-10-08  1.0.6     SY       Update due to the latest introduction of Gym 0.26
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.6 (2022-10-08)

This module shows how to wrap a native MLPro environment class to OpenAI Gym environment.
"""


from mlpro.bf.various import Log
from mlpro.wrappers.openai_gym import WrEnvMLPro2GYM
from mlpro.rl.pool.envs.gridworld import GridWorld
from gym.utils.env_checker import check_env
import gym
import numpy as np


class MyWrEnvMLPro2GYM(WrEnvMLPro2GYM):
    def reset(self, seed=None, return_info=False, options=None):
        """
        We redefine this method because check_env method is only compatible until Gym 0.25.0.
        Meanwhile, our wrapper follow Gym 0.26.2.
        Therefore this class is temporary until check_env method is updated by the developer.

        """
        super().reset(seed=seed)
        
        self._mlpro_env.reset(seed)
        obs = None
        if isinstance(self.observation_space, gym.spaces.Box):
            obs = np.array(self._mlpro_env.get_state().get_values(), dtype=np.float32)
        else:
            obs = np.array(self._mlpro_env.get_state().get_values())
        
        info = {}
        if return_info:
            return obs, info
        else:
            return obs
            

mlpro_env   = GridWorld(p_logging=Log.C_LOG_ALL)
env         = MyWrEnvMLPro2GYM(mlpro_env, p_state_space=None, p_action_space=None, p_new_step_api=True)
check_env(env)
