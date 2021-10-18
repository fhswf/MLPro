## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : test_buffer
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-27  1.0.0     WB       Creation
## -- 2021-09-27  1.0.0     WB       Release First Version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-09-27)

Unit test classes for SARBuffer.
"""


import pytest
import random
import numpy as np
from mlpro.bf.various import *
from mlpro.bf.math import *
from mlpro.bf.ml import *
from mlpro.rl.models import *
from mlpro.rl.pool.envs.robotinhtm import RobotHTM
from mlpro.rl.pool.policies.sac import SAC
from mlpro.rl.pool.sarsbuffer.PrioritizedBuffer import PrioritizedBuffer
from mlpro.rl.pool.sarsbuffer.RandomSARSBuffer import RandomSARSBuffer
import gym
import random
from pathlib import Path



            
## -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("buffer_cls", [PrioritizedBuffer, RandomSARSBuffer])
def test_buffer(buffer_cls):    
    class MyScenario(Scenario):

        C_NAME      = 'Matrix'

        def _setup(self, p_mode, p_ada, p_logging):
            self._env   = RobotHTM(p_logging=False) 

            class SACB(SAC):
                C_BUFFER_CLS = buffer_cls
                
            # 2 Setup standard single-agent with own policy
            self._agent = Agent(
                p_policy=SACB(
                    p_observation_space=self._env.get_state_space(),
                    p_action_space=self._env.get_action_space(),
                    p_batch_size=10,
                    p_buffer_size=10,
                    p_ada=p_ada,
                    p_logging=p_logging
                ),    
                p_envmodel=None,
                p_name='Smith',
                p_ada=p_ada,
                p_logging=p_logging
            )
            
    myscenario  = MyScenario(
        p_mode=Environment.C_MODE_SIM,
        p_ada=True,
        p_cycle_limit=10,
        p_visualize=True,
        p_logging=False,
    )
    training        = Training(
        p_scenario=myscenario,
        p_episode_limit=10,
        p_cycle_limit=10,
        p_collect_states=True,
        p_collect_actions=True,
        p_collect_rewards=True,
        p_collect_training=True,
        p_logging=False
    )

    training.run()
    

