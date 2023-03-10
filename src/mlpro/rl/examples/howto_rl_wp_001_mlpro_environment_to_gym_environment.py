## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_wp_001_mlpro_environment_to_gym_environment.py
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
## -- 2022-10-14  1.0.6     SY       Refactoring 
## -- 2022-11-02  1.0.7     SY       Unable logging in unit test model
## -- 2023-03-02  1.0.8     LSB      Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.8 (2023-03-02)

This module shows how to wrap a native MLPro environment class to OpenAI Gym environment.

You will learn:

1. How to setup an MLPro environment.

2. How to wrap MLPro's native Environment class to the Gym environment object.
"""


from mlpro.bf.various import Log
from mlpro.wrappers.openai_gym import WrEnvMLPro2GYM
from mlpro.rl.pool.envs.gridworld import GridWorld
from gym.utils.env_checker import check_env


if __name__ == "__main__":
    logging = Log.C_LOG_ALL
else:
    logging = Log.C_LOG_NOTHING
    
# 1. Set up MLPro native environment
mlpro_env = GridWorld(p_logging=logging)

# 2. Wrap the MLPro environment to gym compatible environment
env = WrEnvMLPro2GYM(mlpro_env,
                     p_state_space=None,
                     p_action_space=None,
                     p_new_step_api=False,
                     p_logging=logging)

# 3. Check whether the environment is valid
check_env(env)
