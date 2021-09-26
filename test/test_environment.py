## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : test_environment
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-11  1.0.0     MRD      Creation
## -- 2021-09-11  1.0.0     MRD      Release First Version
## -- 2021-09-22  1.0.1     WB       Change Environment Instantiation Method
## -- 2021-09-26  1.0.2     MRD      Change the structure to work with GitHub Automated Test
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2021-09-22)

Unit test classes for environment.
"""


import pytest
import random
import numpy as np
from mlpro.rl.models import *
from mlpro.rl.pool.envs.robotinhtm import RobotHTM
from mlpro.rl.pool.envs.bglp import BGLP
from mlpro.rl.pool.envs.gridworld import GridWorld
# from mlpro.rl.pool.envs.ur5jointcontrol import UR5JointControl


## -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("env_cls", [RobotHTM, BGLP, GridWorld])
def test_environment(env_cls):
    assertIsInstance(env_cls, Environment)
    
    assert isinstance(env_cls.get_state_space(), ESpace)
    assert env_cls.get_state_space().get_num_dim() != 0
    
    assert isinstance(env_cls.get_action_space(), ESpace)
    assert env_cls.get_action_space().get_num_dim() != 0
    
    state = env_cls.get_state()
    
    assert isinstance(state, State)
        
    my_action_values = np.zeros(env_cls.get_action_space().get_num_dim())
    for d in range(env_cls.get_action_space().get_num_dim()):
        my_action_values[d] = random.random() 

    my_action_values = Action(0, env_cls.get_action_space(), my_action_values)

    env_cls.process_action(my_action_values)

    reward = env_cls.compute_reward()
    
    assert isinstance(reward, Reward)

    env_cls._evaluate_state()

    env_cls.reset()
