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
from mlpro.wrappers.openai_gym import WrEnvGYM2MLPro
from mlpro.rl.pool.sarsbuffer.PrioritizedBuffer import PrioritizedBuffer
from mlpro.rl.pool.sarsbuffer.RandomSARSBuffer import RandomSARSBuffer
from mlpro.rl.pool.policies.dummy import MyDummyPolicy
import gym
import random
from pathlib import Path



            
# -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("buffer_cls", [PrioritizedBuffer, RandomSARSBuffer])
def test_buffer(buffer_cls):    
    class MyScenario(RLScenario):

        C_NAME      = 'Matrix'

        def _setup(self, p_mode, p_ada, p_logging):
            gym_env     = gym.make('CartPole-v1')
            self._env   = WrEnvGYM2MLPro(gym_env, p_logging=False)

            class MyDummyPol(MyDummyPolicy):
                C_BUFFER_CLS = buffer_cls
                
            # 2 Setup standard single-agent with own policy
            return Agent(
                p_policy=MyDummyPol(
                    p_observation_space=self._env.get_state_space(),
                    p_action_space=self._env.get_action_space(),
                    p_buffer_size=100,
                    p_ada=p_ada,
                    p_logging=p_logging
                ),    
                p_envmodel=None,
                p_name='Smith',
                p_ada=p_ada,
                p_logging=p_logging
            )
            
    training        = RLTraining(
        p_scenario_cls=MyScenario,
        p_cycle_limit=100,
        p_max_stagnations=0,
        p_collect_states=True,
        p_collect_actions=True,
        p_collect_rewards=True,
        p_collect_training=True,
        p_logging=False
    )

    training.run()
    

