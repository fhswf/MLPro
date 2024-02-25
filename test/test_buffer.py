## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : test_buffer.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-27  1.0.0     WB       Creation
## -- 2021-09-27  1.0.0     WB       Release First Version
## -- 2022-11-07  1.1.0     DA       Refactoring
## -- 2023-04-19  1.1.1     MRD      Refactor module import gym to gymnasium
## -- 2024-02-16  1.1.2     SY       Replace gym environment to BGLP to remove dependency
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.2 (2023-04-19)

Unit test classes for SARBuffer.
"""


import pytest
from mlpro.bf.various import *
from mlpro.bf.math import *
from mlpro.bf.ml import *
from mlpro.rl.models import *
from mlpro.rl.pool.envs.bglp import BGLP
from mlpro.rl.pool.sarsbuffer.PrioritizedBuffer import PrioritizedBuffer
from mlpro.rl.pool.sarsbuffer.RandomSARSBuffer import RandomSARSBuffer
from mlpro.rl.pool.policies.dummy import MyDummyPolicy



            
# -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("buffer_cls", [PrioritizedBuffer, RandomSARSBuffer])
def test_buffer(buffer_cls):    
    class MyScenario(RLScenario):

        C_NAME      = 'Matrix'

        def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
            self._env = BGLP(p_logging=p_logging, cycle_limit=100)

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
        p_visualize=False,
        p_logging=Log.C_LOG_NOTHING
    )

    training.run()
    

