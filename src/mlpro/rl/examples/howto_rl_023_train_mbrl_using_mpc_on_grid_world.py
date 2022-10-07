## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : howto_rl_023_train_mbrl_using_mpc_on_grid_world.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-09-19  0.0.0     SY       Creation
## -- 2022-10-06  1.0.0     SY       Release first version
## -------------------------------------------------------------------------------------------------


"""
Ver. 1.0.0 (2022-10-06)

This module shows how to incorporate MPC in Model-Based RL on Grid World problem.

You will learn:
    
1) How to set up an own agent on Grid World problem
    
2) How to set up model-based RL (MBRL) training
    
3) How to incorporate MPC into MBRL training
"""

from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.pool.envs.gridworld import *
from mlpro.rl.pool.policies.randomgenerator import RandomGenerator
from pathlib import Path
from mlpro.rl.pool.actionplanner.mpc import MPC
from mlpro.rl.pool.envmodels.mlp_gridworld import MLPEnvModel





# 1 Implement the random RL scenario
class ScenarioGridWorld(RLScenario):

    C_NAME      = 'Grid World with Random Actions'

    def _setup(self, p_mode, p_ada, p_logging):
        # 1.1 Setup environment
        self._env   = GridWorld(p_logging=p_logging,
                                p_action_type=GridWorld.C_ACTION_TYPE_DISC_2D,
                                p_max_step=100)


        # 1.2 Setup and return random action agent
        policy_random = RandomGenerator(p_observation_space=self._env.get_state_space(), 
                                        p_action_space=self._env.get_action_space(),
                                        p_buffer_size=1,
                                        p_ada=1,
                                        p_logging=p_logging)

        mb_training_param = dict(p_cycle_limit=100,
                                 p_cycles_per_epi_limit=100,
                                 p_max_stagnations=0,
                                 p_collect_states=False,
                                 p_collect_actions=False,
                                 p_collect_rewards=False,
                                 p_collect_training=False)

        return Agent(
            p_policy=policy_random,  
            p_envmodel=MLPEnvModel(),
            p_em_acc_thsld=0.2,
            p_action_planner=MPC(),
            p_predicting_horizon=5,
            p_controlling_horizon=1,
            p_planning_width=50,
            p_name='Smith',
            p_ada=p_ada,
            p_logging=p_logging,
            **mb_training_param
        )



# 2 Train agent in scenario
if __name__ == "__main__":
    # 2.1 Parameters for demo mode
    cycle_limit = 50000
    logging     = Log.C_LOG_ALL
    visualize   = True
    path        = str(Path.home())
    plotting    = True
else:
    # 2.2 Parameters for internal unit test
    cycle_limit = 100
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    path        = None
    plotting    = False

training = RLTraining(
    p_scenario_cls=ScenarioGridWorld,
    p_cycle_limit=cycle_limit,
    p_cycles_per_epi_limit=100,
    p_collect_states=True,
    p_collect_actions=True,
    p_collect_rewards=True,
    p_collect_training=True,
    p_path=path,
    p_logging=logging,
)

training.run()