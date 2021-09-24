## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : Howto 07 - Train UR5 Robot environment with A2C Algorithm
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-23  0.0.0     WB       Creation
## -- 2021-09-23  1.0.0     WB       Released first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-09-23)

This module shows how to implement A2C on the UR5 Robot Environment
"""


from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.pool.envs.ur5jointcontrol import UR5JointControl
from mlpro.rl.pool.policies.a2c import A2C 
import random
from pathlib import Path

# 1 Make Sure training_env branch of ur_control is sourced:
    # request access to the ur_control project

# 2 Implement your own RL scenario
class ScenarioUR5A2C(Scenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):
        # 1 Setup environment
        self._env   = UR5JointControl(p_logging=True) 

        # 2 Setup standard single-agent with own policy
        self._agent = Agent(
            p_policy=A2C(
                p_state_space=self._env.get_state_space(),
                p_action_space=self._env.get_action_space(),
                p_buffer_size=100,
                p_ada=p_ada,
                p_logging=p_logging
            ),    
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_logging=p_logging
        )




# 3 Instantiate scenario
myscenario  = ScenarioUR5A2C(
    p_mode=Environment.C_MODE_SIM,
    p_ada=True,
    p_cycle_limit=100,
    p_visualize=True,
    p_logging=False
)




# 4 Train agent in scenario 
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

# 5 Saving
ts              = '%04d-%02d-%02d  %02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
dest_path       = str(Path.home()) + os.sep + 'ccb rl - howto 07' + os.sep + ts
