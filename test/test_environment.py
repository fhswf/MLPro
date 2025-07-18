## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : test_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-11  1.0.0     MRD      Creation
## -- 2021-09-11  1.0.0     MRD      Release First Version
## -- 2021-09-22  1.0.1     WB       Change Environment Instantiation Method
## -- 2021-09-26  1.0.2     MRD      Change the structure to work with GitHub Automated Test
## -- 2021-09-26  1.0.3     MRD      Change the import module due to the change of the pool
## --                                folder structer
## -- 2021-09-26  1.0.4     MRD      Change the import module due to the change of the pool
## --                                folder structer
## -- 2022-09-02  1.0.5     SY       Add DoublePendulumS7 and DoublePendulumS4
## -- 2022-09-13  1.0.5     SY       Add Sim_MPPS
## -- 2022-11-22  1.0.6     SY       Remove Sim_MPPS
## -- 2024-02-16  1.0.7     SY       Remove Multi-Cartpole
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.7 (2024-02-16)

Unit test classes for environment.
"""


import pytest
import random
import numpy as np

from mlpro.bf.math import ESpace
from mlpro.bf.systems import State, Action

from mlpro.rl.models import *
from mlpro.rl.pool.envs.robotinhtm import RobotHTM
from mlpro.rl.pool.envs.bglp import BGLP
from mlpro.rl.pool.envs.gridworld import GridWorld
from mlpro.rl.pool.envs.doublependulum import DoublePendulumS4, DoublePendulumS7


## -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("env_cls", [RobotHTM, BGLP, GridWorld, DoublePendulumS7, DoublePendulumS4])
def test_environment(env_cls):
    env = env_cls(p_visualize=False)
    assert isinstance(env, Environment)
    
    assert isinstance(env.get_state_space(), ESpace)
    assert env.get_state_space().get_num_dim() != 0
    
    assert isinstance(env.get_action_space(), ESpace)
    assert env.get_action_space().get_num_dim() != 0
    
    state = env.get_state()
    
    assert isinstance(state, State)
        
    my_action_values = np.zeros(env.get_action_space().get_num_dim())
    for d in range(env.get_action_space().get_num_dim()):
        my_action_values[d] = random.random() 

    my_action_values = Action(0, env.get_action_space(), my_action_values)

    env.process_action(my_action_values)

    reward = env.compute_reward()
    
    assert isinstance(reward, Reward)

    env.reset()
