## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : Howto-RL-008_Wrap_MLPro_environment_to_Gym_environment.py 
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-30  0.0.0     SY       Creation
## -- 2021-09-30  1.0.0     SY       Released first version
## -- 2021-10-04  1.0.1     DA       Minor fixes
## -- 2021-12-22  1.0.2     DA       Cleaned up a bit
## -- 2022-03-21  1.0.3     MRD      Use Gym Env Checker
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.3 (2022-03-21)

This module shows how to wrap mlpro's Environment class to gym compatible.
"""


from mlpro.wrappers.openai_gym import WrEnvMLPro2GYM
from mlpro.rl.pool.envs.gridworld import GridWorld
from gym.utils.env_checker import check_env



mlpro_env   = GridWorld(p_logging=True)
env         = WrEnvMLPro2GYM(mlpro_env, p_state_space=None, p_action_space=None)
check_env(env)
