## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_wp_004_train_agent_with_sb3_policy.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-29  0.0.0     MRD      Creation
## -- 2021-10-07  1.0.0     MRD      Released first version
## -- 2021-10-08  1.0.1     DA       Take over the cycle limit from the environment
## -- 2021-10-18  1.0.2     DA       Refactoring
## -- 2021-10-18  1.0.3     MRD      SB3 Off Policy Wrapper DQN, DDPG, SAC
## -- 2021-11-15  1.0.4     DA       Refactoring
## -- 2021-12-03  1.0.5     DA       Refactoring
## -- 2021-12-07  1.0.6     DA       Refactoring
## -- 2022-02-25  1.0.7     SY       Refactoring due to auto generated ID in class Dimension
## -- 2022-07-20  1.0.8     SY       Update due to the latest introduction of Gym 0.25
## -- 2022-10-14  1.0.9     SY       Refactoring 
## -- 2022-11-07  1.1.0     DA       Refactoring 
## -- 2023-01-14  1.1.1     MRD      Removing default parameter new_step_api and render_mode for gym
## -- 2023-02-13  1.1.2     DA       Optimization of dark mode
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.2 (2023-02-13)

This module shows how to train agent with SB3 Wrapper for On- and Off-Policy Algorithms

You will learn:
    
1) How to set up a scenario with SB3 policy

2) How to run the scenario
    
"""


import gym
from stable_baselines3 import A2C, PPO, DQN, DDPG, SAC
from mlpro.rl.models import *
from mlpro.wrappers.openai_gym import WrEnvGYM2MLPro
from mlpro.wrappers.sb3 import WrPolicySB32MLPro
from collections import deque
from pathlib import Path


# 1 Implement your own RL scenario
class MyScenario(RLScenario):
    C_NAME = 'Howto-RL-WP-004'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        # 1 Setup environment
        # self._env   = RobotHTM(p_logging=False)
        gym_env = gym.make('CartPole-v1')
        self._env = WrEnvGYM2MLPro(gym_env, p_visualize=p_visualize, p_logging=p_logging)

        # 2 Instantiate Policy From SB3
        # env is set to None, it will be set up later inside the wrapper
        # _init_setup_model is set to False, the _setup_model() will be called inside
        # the wrapper manually

        # A2C
        # policy_sb3 = A2C(
        #             policy="MlpPolicy", 
        #             env=None,
        #             use_rms_prop=False, 
        #             _init_setup_model=False,
        #             device="cpu")

        # PPO
        policy_sb3 = PPO(
            policy="MlpPolicy",
            n_steps=5,
            env=None,
            _init_setup_model=False,
            device="cpu")

        # DQN Discrete only
        # policy_sb3 = DQN(
        #             policy="MlpPolicy", 
        #             env=None,
        #             _init_setup_model=False,
        #             device="cpu")

        # DDPG Continuous only
        # policy_sb3 = DDPG(
        #             policy="MlpPolicy", 
        #             env=None,
        #             _init_setup_model=False,
        #             device="cpu")

        # SAC Continuous only
        # policy_sb3 = SAC(
        #             policy="MlpPolicy", 
        #             env=None,
        #             _init_setup_model=False,
        #             device="cpu")

        # 3 Wrap the policy
        policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=policy_sb3,
            p_cycle_limit=self._cycle_limit,
            p_observation_space=self._env.get_state_space(),
            p_action_space=self._env.get_action_space(),
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging)

        # 4 Setup standard single-agent with own policy
        return Agent(
            p_policy=policy_wrapped,
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging
        )


# 2 Create scenario and start training
if __name__ == "__main__":
    # 2.1 Parameters for demo mode
    logging = Log.C_LOG_ALL
    cycle_limit = 1000
    visualize = True
    path = str(Path.home())

else:
    # 2.2 Parameters for internal unit test
    logging = Log.C_LOG_NOTHING
    cycle_limit = 50
    visualize = False
    path = None

# 2.3 Create and run training object
training = RLTraining(
    p_scenario_cls=MyScenario,
    p_cycle_limit=cycle_limit,
    p_max_adaptations=0,
    p_max_stagnations=0,
    p_path=path,
    p_visualize=visualize,
    p_logging=logging)

training.run()
