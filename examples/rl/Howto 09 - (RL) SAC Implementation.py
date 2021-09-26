## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : Howto 09 - SAC Implementation
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-24  0.0.0     MRD      Creation
## -- 2021-09-25  1.0.0     MRD      Released first version
## -- 2021 09-26  1.0.1     MRD      Change the import module due to the change of the pool
## --                                folder structer
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2021-09-26)

This module shows how to implement SAC from the pool
"""


from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.wrappers import WrEnvGym
from mlpro.rl.pool.envs import RobotHTM
from mlpro.rl.pool.policies import SAC
import gym
import random
from pathlib import Path

# 1 Implement your own RL scenario
class MyScenario(Scenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):
        # 1 Setup environment
        self._env   = RobotHTM(p_logging=False)
        # gym_env     = gym.make('CartPole-v1')
        # gym_env     = gym.make('MountainCarContinuous-v0')
        # self._env   = WrEnvGym(gym_env, p_logging=False) 

        # 2 Setup standard single-agent with own policy
        self._agent = Agent(
            p_policy=SAC(
                p_state_space=self._env.get_state_space(),
                p_action_space=self._env.get_action_space(),
                p_buffer_size=1000000,
                p_ada=p_ada,
                p_logging=p_logging
            ),    
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_logging=p_logging
        )




# 2 Instantiate scenario
myscenario  = MyScenario(
    p_mode=Environment.C_MODE_SIM,
    p_ada=True,
    p_cycle_limit=100,
    p_visualize=True,
    p_logging=False
)




# 3 Train agent in scenario 
now             = datetime.now()

training        = Training(
    p_scenario=myscenario,
    p_episode_limit=2000,
    p_cycle_limit=100,
    p_collect_states=True,
    p_collect_actions=True,
    p_collect_rewards=True,
    p_collect_training=True,
    p_logging=True
)

training.run()

# 4 Saving
ts              = '%04d-%02d-%02d  %02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
dest_path       = str(Path.home()) + os.sep + 'ccb rl - howto 06' + os.sep + ts
