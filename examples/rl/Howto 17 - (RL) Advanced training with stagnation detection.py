## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : Howto 17 - Advanced training with stagnation detection
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-11  0.0.0     DA       Creation
## -- 2021-12-12  1.0.0     DA       Released first version
## -- 2022-02-04  1.1.0     DA       Introduction of parameter p_stagnation_entry
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2022-02-04)

This module demonstrates advanced training with evaluation and stagnation detection.
"""


from mlpro.rl.models import *
from mlpro.rl.pool.envs.multicartpole import MultiCartPole
import random
from pathlib import Path




# 1 Implement your own agent policy
class MyPolicy (Policy):

    C_NAME      = 'MyPolicy'

    def set_random_seed(self, p_seed=None):
        random.seed(p_seed)


    def compute_action(self, p_state: State) -> Action:
        # 1 Create a numpy array for your action values 
        my_action_values = np.zeros(self._action_space.get_num_dim())

        # 2 Computing action values is up to you...
        for d in range(self._action_space.get_num_dim()):
            my_action_values[d] = random.random() 

        # 3 Return an action object with your values
        return Action(self._id, self._action_space, my_action_values)


    def _adapt(self, *p_args) -> bool:
        # 1 Adapting the internal policy is up to you...
        self.log(self.C_LOG_TYPE_I, 'Sorry, I am a stupid agent...')

        # 2 Only return True if something has been adapted...
        return False




# 2 Implement your own RL scenario
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
        multi_agent.add_agent(
            p_agent=Agent(
                p_policy=MyPolicy(
                    p_observation_space=self._env.get_state_space().spawn([0,1,2,3]),
                    p_action_space=self._env.get_action_space().spawn([0]),
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
                p_policy=MyPolicy(
                    p_observation_space=self._env.get_state_space().spawn([4,5,6,7,8,9,10,11]),
                    p_action_space=self._env.get_action_space().spawn([1,2]),
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
    cycle_limit         = 5000
    adaptation_limit    = 50
    stagnation_limit    = 5
    stagnation_entry    = 3
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
        p_score_ma_horizon=3,
        p_success_ends_epi=True,
        p_path=path,
        p_visualize=visualize,
        p_logging=logging )

training.run()