## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto_rl_020_run_double_pendulum_with_random_actions.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-04-23  0.0.0     YI       Creation
## -- 2022-04-28  0.0.0     YI       Changing the Scenario and Debugging
## -- 2022-05-16  1.0.0     SY       Code cleaning, remove unnecessary, release the first version
## -- 2022-06-21  1.0.1     SY       Adjust the name of the module, utilize RandomGenerator class
## -- 2022-08-02  1.0.2     LSB      Parameters for internal unit testing
## -- 2022-08-05  1.0.3     SY       Refactoring
## -------------------------------------------------------------------------------------------------


"""
Ver. 1.0.3 (2022-08-05)

This module shows how to use run the double pendulum environment using random actions agent
"""

from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.pool.envs.doublependulum import *
from mlpro.rl.pool.policies.randomgenerator import RandomGenerator
from pathlib import Path
import matplotlib.pyplot as plt




# This command is required for some IDEs to generate the adaptive plot
plt.ion()

# 1 Implement the random RL scenario
class ScenarioDoublePendulum(RLScenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):
        # 1.1 Setup environment
        self._env   = DoublePendulum_bak(p_logging=True, init_angles='up', max_torque=5)
        # policy_kwargs = dict(activation_fn=torch.nn.Tanh,
        #              net_arch=[dict(pi=[128, 128], vf=[128, 128])])

        # 1.2 Setup random action generator
        policy_random = RandomGenerator(p_observation_space=self._env.get_state_space(), 
                                        p_action_space=self._env.get_action_space(),
                                        p_buffer_size=1,
                                        p_ada=1,
                                        p_logging=False)

        return Agent(
            p_policy=policy_random,  
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_logging=p_logging
        )



# 2 Create scenario and start training
if __name__ == "__main__":
    # 2.1 Parameters for demo mode
    cycle_limit         = 200
    adaptation_limit    = 10000
    stagnation_limit    = 0
    eval_frequency      = 5
    eval_grp_size       = 5
    logging             = Log.C_LOG_WE
    visualize           = True
    path                = str(Path.home())
    plotting            = True
else:
    # 2.2 Parameters for unittest
    cycle_limit         = 20
    adaptation_limit    = 10
    stagnation_limit    = 0
    eval_frequency      = 5
    eval_grp_size       = 5
    logging             = Log.C_LOG_NOTHING
    visualize           = False
    path                = None
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




