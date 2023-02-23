## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_bf_systems_xxx_double_pendulum_as_system.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-mm-dd  0.0.0     LSB       Creation
## -------------------------------------------------------------------------------------------------


"""
Ver. 1.2.0 (2023-02-23)

This module shows how to run the double pendulum environment using random actions agent.

You will learn:

1) ...

"""


from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.pool.envs.doublependulum import *
from mlpro.rl.pool.policies.randomgenerator import RandomGenerator
from pathlib import Path
from mlpro.bf.systems.scenario import SystemScenario
import numpy as np



# 1 Implement the random RL scenario
class ScenarioDoublePendulum(SystemScenario):

    C_NAME      = 'Double Pendulum with Random Actions'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        # 1.1 Setup environment
        self._system = DoublePendulumRoot(p_latency=timedelta(0,0,40000),p_init_angles='random', p_max_torque=10,
            p_visualize=p_visualize,
            p_logging=p_logging)

        self._operation_mode = self.C_OP_RND
        # 1.2 Setup and return random action agent
        policy_random = RandomGenerator(p_observation_space=self._system.get_state_space(),
                                        p_action_space=self._system.get_action_space(),
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



# 2 Create scenario and run the scenario
if __name__ == "__main__":
    # 2.1 Parameters for demo mode
    cycle_limit         = 300
    logging             = Log.C_LOG_ALL
    visualize           = True
    plotting            = True
else:
    # 2.2 Parameters for unittest
    cycle_limit         = 2
    logging             = Log.C_LOG_NOTHING
    visualize           = False
    plotting            = False



# 3 Create your scenario and run some cycles
myscenario  = ScenarioDoublePendulum(
    p_mode=Mode.C_MODE_SIM,
    p_ada=True,
    p_cycle_limit=cycle_limit,
    p_visualize=visualize,
    p_logging=logging
)

myscenario.reset(p_seed=3)
myscenario.run()

