## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto_rl_009_wrap_mlpro_environment_to_pettingzoo_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-10-02  0.0.0     SY       Creation
## -- 2021-10-02  1.0.0     SY       Released first version
## -- 2021-10-04  1.0.1     DA       Minor fix
## -- 2021-11-15  1.0.2     DA       Refactoring
## -- 2021-12-03  1.0.3     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.3 (2021-12-03)

This module shows how to wrap mlpro's Environment class to petting zoo compatible.
"""


from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.wrappers.pettingzoo import WrEnvMLPro2PZoo
from mlpro.rl.pool.envs.bglp import BGLP
import random

from pettingzoo.test import api_test

mlpro_env   = BGLP(p_logging=Mode.C_LOG_ALL)
env         = WrEnvMLPro2PZoo(mlpro_env, p_num_agents=5, p_state_space=None, p_action_space=None).pzoo_env
try:
    api_test(env, num_cycles=10, verbose_progress=False)
    print("test completed")
    assert True
except:
    print("test failed")  
    assert False
    
