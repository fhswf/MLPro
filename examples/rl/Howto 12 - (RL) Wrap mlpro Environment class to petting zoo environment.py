## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : Howto 12 - (RL) Wrap mlpro Environment class to petting zoo environment
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-10-02  0.0.0     SY       Creation
## -- 2021-10-02  1.0.0     SY       Released first version
## -- 2021-10-04  1.0.1     DA       Minor fix
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2021-10-04)

This module shows how to wrap mlpro's Environment class to petting zoo compatible.
"""


from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.wrappers import WrEnvMLPro2PZoo
from mlpro.rl.pool.envs.bglp import BGLP
import random

from pettingzoo.test import api_test

mlpro_env   = BGLP(p_logging=True)
env         = WrEnvMLPro2PZoo(mlpro_env, p_num_agents=5, p_state_space=None, p_action_space=None).pzoo_env
try:
    api_test(env, num_cycles=10, verbose_progress=False)
    print("test completed")
except:
    print("test failed")     
