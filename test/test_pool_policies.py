## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : test_pool_policies
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-08  1.0.0     MRD      Creation
## -- 2022-05-19  1.0.1     SY       Add RandomGenerator policy
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-12-08)

Unit test classes for policies.
"""


import pytest
import random
import gym
import numpy as np
from mlpro.rl.models import *
from mlpro.wrappers.openai_gym import WrEnvGYM2MLPro
from mlpro.rl.pool.policies.dummy import MyDummyPolicy
from mlpro.rl.pool.policies.randomgenerator import RandomGenerator

## Instructions
# Please Include your own policy implementation class on the test list parameter


## -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("policy_cls", [MyDummyPolicy, RandomGenerator])
def test_pool_policies(policy_cls):
    class MyScenario (RLScenario):

        C_NAME      = 'Matrix'

        def _setup(self, p_mode, p_ada, p_logging):
            # 1 Setup environment
            gym_env     = gym.make('CartPole-v1')
            self._env   = WrEnvGYM2MLPro(gym_env, p_logging=False) 

            # 2 Setup and return standard single-agent with own policy
            return Agent(
                p_policy=policy_cls(
                    p_observation_space=self._env.get_state_space(),
                    p_action_space=self._env.get_action_space(),
                    p_buffer_size=10,
                    p_ada=p_ada,
                    p_logging=p_logging
                ),    
                p_envmodel=None,
                p_name='Smith',
                p_ada=p_ada,
                p_logging=p_logging
            )

    # 2.4 Create and run training object
    training = RLTraining(
            p_scenario_cls=MyScenario,
            p_cycle_limit=100,
            p_max_adaptations=0,
            p_max_stagnations=0,
            p_logging=False
    )

    training.run()