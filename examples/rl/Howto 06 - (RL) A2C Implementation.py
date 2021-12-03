## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : Howto 06 - Train using A2C from the pool
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-13  0.0.0     MRD      Creation
## -- 2021-09-18  1.0.0     MRD      Released first version
## -- 2021 09-26  1.0.1     MRD      Change the import module due to the change of the pool
## --                                folder structer
## -- 2021-09-29  1.0.2     SY       Change name: WrEnvGym to WrEnvGYM2MLPro
## -- 2021-10-18  1.0.3     DA       Refactoring
## -- 2021-11-15  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2021-11-15)

This module shows how to implement A2C from the pool
"""


from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.wrappers.openai_gym import WrEnvGYM2MLPro
from mlpro.rl.pool.envs.robotinhtm import RobotHTM
from mlpro.rl.pool.policies.a2c import A2C 
import gym
import random
from pathlib import Path



# 1 Implement your own RL scenario
class MyScenario (RLScenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):
        # 1 Setup environment
        # self._env   = RobotHTM(p_logging=False)
        gym_env     = gym.make('CartPole-v1')
        # gym_env     = gym.make('MountainCarContinuous-v0')
        self._env   = WrEnvGYM2MLPro(gym_env, p_logging=False) 

        # 2 Setup and return standard single-agent with own policy
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




# 2 Create scenario and start training

if __name__ == "__main__":
    # 2.1 Parameters for demo mode
    logging     = Log.C_LOG_ALL
    visualize   = True
    path        = str(Path.home())
 
else:
    # 2.2 Parameters for internal unit test
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    path        = None


# 2.3 Create your scenario
myscenario  = MyScenario(
    p_mode=Environment.C_MODE_SIM,
    p_ada=True,
    p_cycle_limit=100,
    p_visualize=visualize,
    p_logging=logging
)


# 2.4 Create and run training object
training = RLTraining(
        p_scenario=myscenario,
        p_cycle_limit=100,
        p_max_adaptations=0,
        p_max_stagnations=0,
        p_path=path,
        p_logging=logging
)

training.run()