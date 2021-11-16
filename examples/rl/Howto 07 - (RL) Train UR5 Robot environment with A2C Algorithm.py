## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : Howto 07 - Train UR5 Robot environment with A2C Algorithm
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-23  0.0.0     WB       Creation
## -- 2021-09-23  1.0.0     WB       Released first version
## -- 2021 09-26  1.0.1     MRD      Change the import module due to the change of the pool
## --                                folder structure
## -- 2021-10-18  1.0.2     DA       Refactoring
## -- 2021-11.15  1.0.3     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.3 (2021-11-15)

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
class ScenarioUR5A2C (RLScenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):
        # 1 Setup environment
        self._env   = UR5JointControl(p_logging=True) 

        # 2 Setup standard single-agent with own policy
        return Agent(
            p_policy=A2C(
                p_observation_space=self._env.get_state_space(),
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




# 3 Create scenario and start training

if __name__ == "__main__":
    # 3.1 Parameters for demo mode
    logging     = Log.C_LOG_ALL
    visualize   = True
    path        = str(Path.home())
 
else:
    # 3.2 Parameters for internal unit test
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    path        = None


# 3.3 Create scenario UR5/A2C
myscenario  = ScenarioUR5A2C(
    p_mode=Mode.C_MODE_SIM,
    p_ada=True,
    p_cycle_limit=100,
    p_visualize=visualize,
    p_logging=logging
)


# 3.4 Create and run training object
training = RLTraining(
        p_scenario=myscenario,
        p_cycle_limit=2000,
        p_max_adaptations=0,
        p_max_stagnations=0,
        p_path=path,
        p_logging=logging
)

training.run()