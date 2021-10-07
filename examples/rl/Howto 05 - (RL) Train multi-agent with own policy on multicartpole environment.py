## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : Howto 05 - (RL) Run multi-agent with own policy on multicartpole environment
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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.2 (2021-10-06)

This module shows how to train an own multi-agent with the enhanced multi-action environment 
MultiCartPole based on the OpenAI Gym CartPole environment.
"""



from mlpro.rl.models import *
from mlpro.rl.pool.envs import MultiCartPole
import random
from pathlib import Path
import os
from datetime import datetime




# 1 Implement your own agent policy
class MyPolicy(Policy):

    C_NAME      = 'MyPolicy'

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
        self.log(self.C_LOG_TYPE_W, 'Sorry, I am a stupid agent...')

        # 3 Only return True if something has been adapted...
        return False




# 2 Implement your own RL scenario
class MyScenario(Scenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):

        # 1 Setup Multi-Agent Environment (consisting of 3 OpenAI Gym Cartpole envs)
        self._env   = MultiCartPole(p_num_envs=3, p_logging=p_logging)


        # 2 Setup Multi-Agent 

        # 2.1 Create empty Multi-Agent
        self._agent     = MultiAgent(
            p_name='Smith',
            p_ada=True,
            p_logging=True
        )

        # 2.2 Add Single-Agent #1 with own policy (controlling sub-environment #1)
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=MyPolicy(
                    p_state_space=self._env.get_state_space().spawn([0,1,2,3]),
                    p_action_space=self._env.get_action_space().spawn([0]),
                    p_buffer_size=1,
                    p_ada=True,
                    p_logging=True
                ),
                p_envmodel=None,
                p_name='Smith-1',
                p_id=0,
                p_ada=True,
                p_logging=True
            ),
            p_weight=0.3
        )


        # 2.2 Add Single-Agent #2 with own policy (controlling sub-environments #2,#3)
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=MyPolicy(
                    p_state_space=self._env.get_state_space().spawn([4,5,6,7,8,9,10,11]),
                    p_action_space=self._env.get_action_space().spawn([1,2]),
                    p_buffer_size=1,
                    p_ada=True,
                    p_logging=True
                ),
                p_envmodel=None,
                p_name='Smith-2',
                p_id=1,
                p_ada=True,
                p_logging=True
            ),
            p_weight=0.7
        )




# 3 Instantiate scenario
myscenario  = MyScenario(
    p_mode=Environment.C_MODE_SIM,
    p_ada=True,
    p_cycle_limit=100,
    p_visualize=False,
    p_logging=True
)




# 4 Train agent in scenario 
now             = datetime.now()

training        = Training(
    p_scenario=myscenario,
    p_episode_limit=1,
    p_cycle_limit=100,
    p_collect_states=True,
    p_collect_actions=True,
    p_collect_rewards=True,
    p_collect_training=True,
    p_logging=True
)

training.run()




# 5 Save training data in user home directory
ts              = '%04d-%02d-%02d  %02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
dest_path       = str(Path.home()) + os.sep + 'ccb rl - howto 05' + os.sep + ts
training.save_data(dest_path, "\t")
