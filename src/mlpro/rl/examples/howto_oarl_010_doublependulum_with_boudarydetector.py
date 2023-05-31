## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_oarl_999_adaptive_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-05-31  1.0.0     LSB      Creation/release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-01-12)

This module is an illustration of howto run the online adaptive double pendulum (OADP) environment.

You will learn:

1. How to set up the OADP environment

2. How to create online adaptive tasks

3. How to add the tasks to the reward workflow of the OADP environment

"""


from mlpro.rl.pool.envs.doublependulum import DoublePendulumOA7
from mlpro.rl import *
from mlpro.oa.streams.tasks import BoundaryDetector, NormalizerMinMax
from mlpro.rl.pool.policies.randomgenerator import RandomGenerator



# Creating the Online Adaptive Double Pendulum Environment
environment = DoublePendulumOA7(p_name = '', p_ada=True, p_visualize=True)

# Creating the Boundary Detector Task
task_bd = BoundaryDetector(p_name='Boundary Detector', p_visualize=True, p_range_max=Range.C_RANGE_NONE)

# Creating the Normalizer Task
task_norm = NormalizerMinMax(p_name='Normalizer', p_visualize=True, p_range_max=Range.C_RANGE_NONE)

# Adding the boundary detector task to the Reward Workflow
environment.add_task_reward(p_task=task_bd)

# Adding the normalizer task to the reward workflow
environment.add_task_reward(p_task=task_norm, p_pred_tasks=[task_bd])






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OADPScenario(RLScenario):


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        # 1.1 Setup environment
        self._env   = environment


        # 1.2 Setup and return random action agent
        policy_random = RandomGenerator(p_observation_space=self._env.get_state_space(),
                                        p_action_space=self._env.get_action_space(),
                                        p_buffer_size=1,
                                        p_ada=True,
                                        p_visualize=True,
                                        p_logging=p_logging)

        return Agent(
            p_policy=policy_random,
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_visualize=True,
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
myscenario  = OADPScenario( p_mode=Mode.C_MODE_SIM,
                            p_ada=True,
                            p_cycle_limit=cycle_limit,
                            p_visualize=visualize,
                            p_logging=logging )

myscenario.reset(p_seed=3)
myscenario.run()