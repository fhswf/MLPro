## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto_rl_014_advanced_training_with_stagnation_detection.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-11  0.0.0     DA       Creation
## -- 2021-12-12  1.0.0     DA       Released first version
## -- 2022-02-04  1.1.0     DA       Introduction of parameter p_stagnation_entry
## -- 2022-02-10  1.2.0     DA       Introduction of parameter p_end_at_stagnation
## -- 2022-02-27  1.2.1     SY       Refactoring due to auto generated ID in class Dimension
## -- 2022-05-19  1.2.2     SY       Utilize RandomGenerator
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.2 (2022-05-19)

This module demonstrates advanced training with evaluation and stagnation detection.
"""


from mlpro.rl.models import *
from mlpro.rl.pool.envs.multicartpole import MultiCartPole
import random
from pathlib import Path
from mlpro.rl.pool.policies.randomgenerator import RandomGenerator




# 1 Implement your own RL scenario
class MyScenario (RLScenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):

        # 1 Setup Multi-Agent Environment (consisting of 3 OpenAI Gym Cartpole envs)
        self._env   = MultiCartPole(p_num_envs=3, p_reward_type=Reward.C_TYPE_EVERY_AGENT, p_logging=p_logging)


        # 2 Setup Multi-Agent 

        # 2.1 Create empty Multi-Agent
        multi_agent = MultiAgent(
            p_name='Smith',
            p_ada=True,
            p_logging=p_logging
        )

        # 2.2 Add Single-Agent #1 with own policy (controlling sub-environment #1)
        ss_ids = self._env.get_state_space().get_dim_ids()
        as_ids = self._env.get_action_space().get_dim_ids()
        multi_agent.add_agent(
            p_agent=Agent(
                p_policy=RandomGenerator(
                    p_observation_space=self._env.get_state_space().spawn([ss_ids[0],ss_ids[1],ss_ids[2],ss_ids[3]]),
                    p_action_space=self._env.get_action_space().spawn([as_ids[0]]),
                    p_buffer_size=1,
                    p_ada=True,
                    p_logging=p_logging
                ),
                p_envmodel=None,
                p_name='Smith-1',
                p_id=0,
                p_ada=True,
                p_logging=p_logging
            ),
            p_weight=0.3
        )

        # 2.3 Add Single-Agent #2 with own policy (controlling sub-environments #2,#3)
        multi_agent.add_agent(
            p_agent=Agent(
                p_policy=RandomGenerator(
                    p_observation_space=self._env.get_state_space().spawn([ss_ids[4],ss_ids[5],ss_ids[6],ss_ids[7],ss_ids[8],ss_ids[9],ss_ids[10],ss_ids[11]]),
                    p_action_space=self._env.get_action_space().spawn([as_ids[1],as_ids[2]]),
                    p_buffer_size=1,
                    p_ada=True,
                    p_logging=p_logging
                ),
                p_envmodel=None,
                p_name='Smith-2',
                p_id=1,
                p_ada=True,
                p_logging=p_logging
            ),
            p_weight=0.7
        )

        # 2.4 Adaptive ML model (here: our multi-agent) is returned
        return multi_agent



# 3 Create scenario and start training

if __name__ == "__main__":
    # 3.1 Parameters for demo mode
    cycle_limit         = 1500
    adaptation_limit    = 50
    stagnation_limit    = 5
    stagnation_entry    = 3
    end_at_stagnation   = True
    eval_frequency      = 10
    eval_grp_size       = 5
    logging             = Log.C_LOG_WE
    visualize           = True
    path                = str(Path.home())
 
else:
    # 3.2 Parameters for internal unit test
    cycle_limit         = 50
    adaptation_limit    = 5
    stagnation_limit    = 5
    stagnation_entry    = 1
    end_at_stagnation   = True
    eval_frequency      = 2
    eval_grp_size       = 1
    logging             = Log.C_LOG_NOTHING
    visualize           = False
    path                = None


# 3.3 Create and run training object
training = RLTraining(
        p_scenario_cls=MyScenario,
        p_cycle_limit=cycle_limit,
        p_adaptation_limit=adaptation_limit,
        p_eval_frequency=eval_frequency,
        p_eval_grp_size=eval_grp_size,
        p_stagnation_limit=stagnation_limit,
        p_stagnation_entry=stagnation_entry,
        p_end_at_stagnation=end_at_stagnation,
        p_score_ma_horizon=3,
        p_success_ends_epi=True,
        p_path=path,
        p_visualize=visualize,
        p_logging=logging )

training.run()