## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : Howto 11 - (RL) Wrap mlpro Environment class to gym environment 
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-30  0.0.0     SY       Creation
## -- 2021-09-30  1.0.0     SY       Released first version
## -- 2021-10-04  1.0.1     DA       Minor fixes
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2021-10-04)

This module shows how to wrap mlpro's Environment class to gym compatible.
"""


from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.wrappers import WrEnvMLPro2GYM, WrEnvGYM2MLPro
from mlpro.rl.pool.envs.gridworld import GridWorld
from mlpro.rl.pool.envs.robotinhtm import RobotHTM
import random
from stable_baselines3.common.env_checker import check_env

import gym

mlpro_env   = GridWorld(p_logging=True)
env         = WrEnvMLPro2GYM(mlpro_env, p_state_space=None, p_action_space=None)
check_env(env)
