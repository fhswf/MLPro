## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : Howto 11 - (RL) Wrap mlpro Environment class to gym environment 
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-30  0.0.0     SY       Creation
## -- 2021-09-30  1.0.0     SY       Released first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-09-30)

This module shows how to wrap mlpro's Environment class to gym compatible.
"""


from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.wrappers import WrEnvMLPro2GYM
from mlpro.rl.pool.envs import GridWorld
import random
from stable_baselines.common.env_checker import check_env

mlpro_env   = GridWorld(p_logging=True)
env         = WrEnvMLPro2GYM(mlpro_env, p_state_space=None, p_action_space=None)
check_env(env)
