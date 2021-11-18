## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : Howto 10 - Train using SB3 Wrapper
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-29  0.0.0     MRD      Creation
## -- 2021-10-07  1.0.0     MRD      Released first version
## -- 2021-10-08  1.0.1     DA       Take over the cycle limit from the environment
## -- 2021-10-18  1.0.2     DA       Refactoring
## -- 2021-10-18  1.0.3     MRD      SB3 Off Policy Wrapper DQN, DDPG, SAC
## -- 2021-11-15  1.0.4     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.4 (2021-11-15)

This module shows how to train with SB3 Wrapper for On-Policy Algorithm
"""

import gym
from stable_baselines3 import A2C, PPO, DQN, DDPG, SAC
from mlpro.rl.models import *
from mlpro.rl.wrappers import WrEnvGYM2MLPro
from mlpro.rl.wrappers import WrPolicySB32MLPro
from collections import deque
from pathlib import Path



# 1 Implement your own RL scenario
class MyScenario(RLScenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):
        # 1 Setup environment
        # self._env   = RobotHTM(p_logging=False)
        gym_env     = gym.make('CartPole-v1')
        self._env   = WrEnvGYM2MLPro(gym_env, p_logging=False) 

        # 2 Instatiate Policy From SB3
        # env is set to None, it will be set up later inside the wrapper
        # _init_setup_model is set to False, the _setup_model() will be called inside
        # the wrapper manually

        # A2C
        # policy_sb3 = A2C(
        #             policy="MlpPolicy", 
        #             env=None,
        #             use_rms_prop=False, 
        #             _init_setup_model=False)

        # PPO
        policy_sb3 = PPO(
                    policy="MlpPolicy",
                    n_steps=5, 
                    env=None,
                    _init_setup_model=False)

        # DQN Discrete only
        # policy_sb3 = DQN(
        #             policy="MlpPolicy", 
        #             env=None,
        #             _init_setup_model=False)

        # DDPG Continuous only
        # policy_sb3 = DDPG(
        #             policy="MlpPolicy", 
        #             env=None,
        #             _init_setup_model=False)

        # SAC Continuous only
        # policy_sb3 = SAC(
        #             policy="MlpPolicy", 
        #             env=None,
        #             _init_setup_model=False)

        # 3 Wrap the policy
        policy_wrapped = WrPolicySB32MLPro(
                p_sb3_policy=policy_sb3, 
                p_observation_space=self._env.get_state_space(),
                p_action_space=self._env.get_action_space(),
                p_ada=p_ada,
                p_logging=p_logging)
        
        # 4 Setup standard single-agent with own policy
        return Agent(
            p_policy=policy_wrapped,   
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
        p_mode=Mode.C_MODE_SIM,
        p_ada=True,
        p_visualize=visualize,
        p_logging=logging 
)


# 2.4 Create and run training object
training = RLTraining(
        p_scenario=myscenario,
        p_cycle_limit=1000,
        p_max_adaptations=0,
        p_max_stagnations=0,
        p_path=path,
        p_logging=logging
)

training.run()