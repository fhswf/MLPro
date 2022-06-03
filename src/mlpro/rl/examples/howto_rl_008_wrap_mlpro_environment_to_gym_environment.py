## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.4 (2022-05-30)

This module shows how to wrap a native MLPro environment class to OpenAI Gym environment.
"""


from mlpro.bf.various import Log
from mlpro.wrappers.openai_gym import WrEnvMLPro2GYM
from mlpro.rl.pool.envs.gridworld import GridWorld
from gym.utils.env_checker import check_env



mlpro_env   = GridWorld(p_logging=Log.C_LOG_ALL)
env         = WrEnvMLPro2GYM(mlpro_env, p_state_space=None, p_action_space=None)
check_env(env)
