## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_agent_004_train_multiagent_with_own_policy_on_multicartpole_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-06-06  0.0.0     DA       Creation
## -- 2021-06-06  1.0.0     DA       Release of first version
## -- 2021-07-01  1.1.0     DA       Extended by data logging/storing (user home directory)
## -- 2021-07-06  1.1.1     SY       Bugfix due to method Training.save_data() update
## -- 2021-08-28  1.2.0     DA       Introduced Policy
## -- 2021-09-11  1.2.0     MRD      Change Header information to match our new library name
## -- 2021 09-26  1.2.1     MRD      Change the import module due to the change of the pool
## --                                folder structer
## -- 2021-10-06  1.2.2     DA       Refactoring 
## -- 2021-11-15  1.3.0     DA       Refactoring 
## -- 2021-12-07  1.3.1     DA       Refactoring 
## -- 2022-02-25  1.3.2     SY       Refactoring due to auto generated ID in class Dimension
## -- 2022-10-13  1.3.3     SY       Refactoring 
## -- 2022-11-01  1.3.4     DA       Refactoring 
## -- 2022-11-02  1.3.5     DA       Refactoring 
## -- 2022-11-07  1.4.0     DA       Refactoring 
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.0 (2022-11-07)

This module shows how to train an own multi-agent with the enhanced multi-action environment 
MultiCartPole based on the OpenAI Gym CartPole environment.

You will learn:
    
1) How to set up a native policy for an agent
    
2) How to set up a multiagent
    
3) How to set up a scenario

4) How to run the scenario and train the agents
    
"""



from mlpro.rl import *
from mlpro.rl.pool.envs.multicartpole import MultiCartPole
import random
from pathlib import Path
import os
from datetime import datetime




# 1 Implement your own agent policy
class MyPolicy(Policy):

    C_NAME      = 'MyPolicy'

    def compute_action(self, p_state: State) -> Action:
        # 1.1 Create a numpy array for your action values 
        my_action_values = np.zeros(self._action_space.get_num_dim())

        # 1.2 Computing action values is up to you...
        for d in range(self._action_space.get_num_dim()):
            my_action_values[d] = random.random() 

        # 1.3 Return an action object with your values
        return Action(self._id, self._action_space, my_action_values)


    def _adapt(self, p_sars_elem:SARSElement) -> bool:
        # 1.4 Adapting the internal policy is up to you...
        self.log(self.C_LOG_TYPE_I, 'Sorry, I am a stupid agent...')

        # 1.5 Only return True if something has been adapted...
        return False




# 2 Implement your own RL scenario
class MyScenario (RLScenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:

        # 2.1 Setup Multi-Agent Environment (consisting of 3 OpenAI Gym Cartpole envs)
        self._env   = MultiCartPole(p_num_envs=3, p_visualize=p_visualize, p_logging=p_logging)


        # 2.2 Setup Multi-Agent 

        # 2.2.1 Create empty Multi-Agent
        multi_agent     = MultiAgent(
            p_name='Smith',
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging
        )

        # 2.2.2 Add Single-Agent #1 with own policy (controlling sub-environment #1)
        ss_ids = self._env.get_state_space().get_dim_ids()
        as_ids = self._env.get_action_space().get_dim_ids()
        multi_agent.add_agent(
            p_agent=Agent(
                p_policy=MyPolicy(
                    p_observation_space=self._env.get_state_space().spawn([ss_ids[0],ss_ids[1],ss_ids[2],ss_ids[3]]),
                    p_action_space=self._env.get_action_space().spawn([as_ids[0]]),
                    p_buffer_size=1,
                    p_ada=p_ada,
                    p_logging=p_logging
                ),
                p_envmodel=None,
                p_name='Smith-1',
                p_id=0,
                p_ada=p_ada,
                p_logging=p_logging
            ),
            p_weight=0.3
        )

        # 2.2.3 Add Single-Agent #2 with own policy (controlling sub-environments #2,#3)
        multi_agent.add_agent(
            p_agent=Agent(
                p_policy=MyPolicy(
                    p_observation_space=self._env.get_state_space().spawn([ss_ids[4],ss_ids[5],ss_ids[6],ss_ids[7],ss_ids[8],ss_ids[9],ss_ids[10],ss_ids[11]]),
                    p_action_space=self._env.get_action_space().spawn([as_ids[1],as_ids[2]]),
                    p_buffer_size=1,
                    p_ada=p_ada,
                    p_visualize=p_visualize,
                    p_logging=p_logging
                ),
                p_envmodel=None,
                p_name='Smith-2',
                p_id=1,
                p_ada=p_ada,
                p_visualize=p_visualize,
                p_logging=p_logging
            ),
            p_weight=0.7
        )

        # 2.3 Adaptive ML model (here: our multi-agent) is returned
        return multi_agent




# 3 Create scenario and start training
if __name__ == '__main__':
    # 3.1 Parameters for demo mode
    cycle_limit = 200
    logging     = Log.C_LOG_WE
    visualize   = True
    path        = str(Path.home())
 
else:
    # 3.2 Parameters for internal unit test
    cycle_limit = 10
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    path        = None


# 3.3 Create and run training object
training = RLTraining(
        p_scenario_cls=MyScenario,
        p_cycle_limit=cycle_limit,
        p_path=path,
        p_visualize=visualize,
        p_logging=logging )

training.run()