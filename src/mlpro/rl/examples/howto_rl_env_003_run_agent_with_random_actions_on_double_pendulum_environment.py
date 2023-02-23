## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_env_003_run_agent_with_random_actions_on_double_pendulum_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-04-23  0.0.0     YI       Creation
## -- 2022-04-28  0.0.0     YI       Changing the Scenario and Debugging
## -- 2022-05-16  1.0.0     SY       Code cleaning, remove unnecessary, release the first version
## -- 2022-06-21  1.0.1     SY       Adjust the name of the module, utilize RandomGenerator class
## -- 2022-08-02  1.0.2     LSB      Parameters for internal unit testing
## -- 2022-08-05  1.0.3     SY       Refactoring
## -- 2022-08-23  1.0.4     DA       Refactoring
## -- 2022-09-06  1.0.5     LSB/DA   Refactoring
## -- 2022-10-13  1.0.6     SY       Refactoring 
## -- 2022-11-07  1.1.0     DA       Refactoring
## -- 2023-02-23  1.2.0     DA       Renamed 
## -------------------------------------------------------------------------------------------------


"""
Ver. 1.2.0 (2023-02-23)

This module shows how to run the double pendulum environment using random actions agent.

You will learn:

1) How to set up an own agent using MLPro's builtin random actions policy

2) How to set up an own RL scenario including your agent and MLPro's double pendulum environment

3) How to reset and run your own scenario

"""


from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.pool.envs.doublependulum import *
from mlpro.rl.pool.policies.randomgenerator import RandomGenerator
from pathlib import Path




# 1 Implement the random RL scenario
class ScenarioDoublePendulum(RLScenario):

    C_NAME      = 'Double Pendulum with Random Actions'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        # 1.1 Setup environment
        self._env   = DoublePendulumS7(p_init_angles='random', p_max_torque=10, p_visualize=p_visualize, p_logging=p_logging)


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

