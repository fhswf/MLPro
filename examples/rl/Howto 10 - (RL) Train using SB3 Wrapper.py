## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : Howto 10 - Train using SB3 Wrapper
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-29  0.0.0     MRD      Creation
## -- 2021-09-30  1.0.0     MRD      Released first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-09-26)

This module shows how to train with SB3 Wrapper for On-Policy Algorithm
"""

import gym
from stable_baselines3 import A2C, PPO
from mlpro.rl.models import *
from mlpro.rl.wrappers import WrEnvGYM2MLPro
from mlpro.rl.wrappers import WrPolicySB32MLPro
from collections import deque

# 1 Implement your own RL scenario
class MyScenario(Scenario):

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
        policy_sb3 = A2C(
                    policy="MlpPolicy", 
                    env=None,
                    learning_rate=3e-4,
                    use_rms_prop=False, 
                    _init_setup_model=False)

        # PPO
        # policy_sb3 = PPO(
        #             policy="MlpPolicy", 
        #             env=None,
        #             learning_rate=3e-4,
        #             _init_setup_model=False)

        # 3 Wrap the policy
        policy_wrapped = WrPolicySB32MLPro(
                p_sb3_policy=policy_sb3, 
                p_state_space=self._env.get_state_space(),
                p_action_space=self._env.get_action_space(),
                p_buffer_size=500,
                p_ada=p_ada,
                p_logging=p_logging)
        
        # 4 Setup standard single-agent with own policy
        self._agent = Agent(
            p_policy=policy_wrapped,   
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_logging=p_logging
        )

# 2 Instantiate scenario
myscenario  = MyScenario(
    p_mode=Environment.C_MODE_SIM,
    p_ada=True,
    p_cycle_limit=500,
    p_visualize=False,
    p_logging=False
)

# 3 Instantiate training
training        = Training(
    p_scenario=myscenario,
    p_episode_limit=2,
    p_cycle_limit=500,
    p_collect_states=True,
    p_collect_actions=True,
    p_collect_rewards=True,
    p_collect_training=True,
    p_logging=True
)

# 4 Train
training.run()