## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_wp_002_mlpro_environment_to_petting_zoo_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-10-02  0.0.0     SY       Creation
## -- 2021-10-02  1.0.0     SY       Released first version
## -- 2021-10-04  1.0.1     DA       Minor fix
## -- 2021-11-15  1.0.2     DA       Refactoring
## -- 2021-12-03  1.0.3     DA       Refactoring
## -- 2022-10-14  1.0.4     SY       Refactoring 
## -- 2022-11-02  1.0.5     SY       Unable logging in unit test model
## -- 2023-03-02  1.0.6     LSB      Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.6 (2023-03-02)

This module shows how to wrap mlpro's Environment class to petting zoo compatible.

1. How to setup an MLPro environment.

2. How to wrap MLPro's native envrionment to a Petting Zoo environment.
"""


from mlpro.bf.various import Log
from mlpro.wrappers.pettingzoo import WrEnvMLPro2PZoo
from mlpro.rl.pool.envs.bglp import BGLP
from pettingzoo.test import api_test


if __name__ == "__main__":
    logging = Log.C_LOG_ALL
else:
    logging = Log.C_LOG_NOTHING

# 1. Set up MLPro native environment
mlpro_env = BGLP(p_logging=logging)

# 2. Wrap the MLPro environment to PettingZoo compatible environment
env = WrEnvMLPro2PZoo(mlpro_env,
                      p_num_agents=5,
                      p_state_space=None,
                      p_action_space=None,
                      p_logging=logging).pzoo_env

# 3. Check whether the environment is valid
try:
    api_test(env, num_cycles=10, verbose_progress=False)
    print("test completed")
    assert True
except:
    print("test failed")  
    assert False
    
