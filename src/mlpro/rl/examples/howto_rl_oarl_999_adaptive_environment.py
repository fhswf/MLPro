## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_oarl_999_adaptive_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-12  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-01-12)
...

You will learn:

1.

2.

3.

"""

from mlpro.rl.pool.envs.doublependulum import DoublePendulumS4, DoublePendulumRoot
from mlpro.rl.models_env_adaptive_environment import *
from mlpro.rl.models import *
from mlpro.rl.pool.policies.randomgenerator import RandomGenerator
from numpy import integrate






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
# 1 Implement the random RL scenario
class ScenarioDoublePendulum(RLScenario):

    C_NAME      = 'Double Pendulum with Random Actions'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        # 1.1 Setup environment
        self._env   = DoublePendulumS4(p_init_angles='random', p_max_torque=10, p_visualize=p_visualize,
            p_logging=p_logging)


        # 1.2 Setup and return random action agent
        policy_random = RandomGenerator(p_observation_space=self._env.get_state_space(),
                                        p_action_space=self._env.get_action_space(),
                                        p_buffer_size=1,
                                        p_ada=1,
                                        p_visualize=p_visualize,
                                        p_logging=p_logging)

        return Agent(
            p_policy=policy_random,
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging
        )


class DPRewardFunction(OAFctReward):

    pass