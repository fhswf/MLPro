## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_003_validate_dp.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-10-10  1.0.0     LSB       Creation/Release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-10-10)

This module is used to validate the dp environment
"""

import torch
from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.pool.envs.doublependulum import *
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from mlpro.wrappers.sb3 import WrPolicySB32MLPro
from mlpro.wrappers.openai_gym import WrEnvMLPro2GYM
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np




if __name__ == '__main__':
    p_input = True
else:
    p_input = False




## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------
class ActionGenerator(Policy):
    """
    Action Generation based on user input
    """

    def set_user_action(self, p_action):
        if p_input:
            self.user_action = np.asarray([p_action])
        else:
           self.user_action = np.zeros(1)

    def compute_action(self, p_state:State):

        return Action(self._id, self._action_space, self.user_action)

    def _adapt(self, **p_kwargs) -> bool:
        self.log(self.C_LOG_TYPE_W, 'Sorry I am not adapting anything')
        return False



# 1 Implement the random RL scenario
class ScenarioDoublePendulum(RLScenario):

    C_NAME      = 'Double Pendulum with Random Actions'

    def _setup(self, p_mode, p_ada, p_visualize,  p_logging):
        self.user_action_cycles = 0

        # 1.1 Setup environment
        self._env   = DoublePendulumS7(p_init_angles='up', p_max_torque=10, p_logging=p_logging)


        # 1.2 Setup and return random action agent
        policy_user = ActionGenerator(p_observation_space=self._env.get_state_space(),
                                        p_action_space=self._env.get_action_space(),
                                        p_buffer_size=1,
                                        p_ada=1,
                                        p_logging=p_logging)

        self.user_action_cycles = 0

        return Agent(
            p_policy=policy_user,
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_logging=p_logging
        )


    def _run_cycle(self):

        if p_input and self.get_cycle_id() == self.user_action_cycles:
            p_torque = int(input('Enter the amount of torque in Nm:'))
            self.get_agent()._policy.set_user_action(p_torque)
            p_cycles = int(input('Enter the amount of cycles to be executed:'))

        elif not p_input:
            self.get_agent()._policy.set_user_action(0)
            p_cycles = 0

        else:
            p_cycles = 0

        self.user_action_cycles += p_cycles
        success, error, adapted, end_of_data = super()._run_cycle()
        return success, error, adapted, end_of_data


# 2 Create scenario and run the scenario
if __name__ == "__main__":
    # 2.1 Parameters for demo mode
    cycle_limit         = 20000
    logging             = Log.C_LOG_ALL
    visualize           = True
    plotting            = True
else:
    # 2.2 Parameters for unittest
    cycle_limit         = 20
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

