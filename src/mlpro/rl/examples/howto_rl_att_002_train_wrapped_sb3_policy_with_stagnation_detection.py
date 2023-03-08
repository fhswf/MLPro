## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_att_002_train_wrapped_sb3_policy_with_stagnation_detection.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-01-20  0.0.0     MRD      Creation
## -- 2022-01-20  1.0.0     MRD      Released first version
## -- 2022-05-17  1.0.1     DA       Just a litte comment maintenance
## -- 2022-07-20  1.0.2     SY       Update due to the latest introduction of Gym 0.25
## -- 2022-10-13  1.0.3     SY       Refactoring 
## -- 2022-10-19  1.0.4     DA       Renamed 
## -- 2022-11-01  1.0.5     DA       Refactoring 
## -- 2022-11-07  1.1.0     DA       Refactoring 
## -- 2023-01-14  1.1.1     MRD      Removing default parameter new_step_api and render_mode for gym
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.1 (2023-01-14)

This module shows how to train with SB3 Wrapper and stagnation detection

You will learn:
    
1) How to incorporate stagnation detection into your training with sb3 policy
    
2) The effect of each parameter related to the stagnation detection
    
"""


import gym
from stable_baselines3 import A2C, PPO, DQN, DDPG, SAC
from mlpro.rl import *
from mlpro.wrappers.openai_gym import WrEnvGYM2MLPro
from mlpro.wrappers.sb3 import WrPolicySB32MLPro
from pathlib import Path



# 1 Implement your own RL scenario
class MyScenario(RLScenario):
    C_NAME = 'Matrix'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:

        # 1 Setup environment
        gym_env = gym.make('CartPole-v1')
        self._env = WrEnvGYM2MLPro(gym_env, p_visualize=p_visualize, p_logging=p_logging)

        # 2 Instantiate PPO Policy from SB3
        policy_sb3 = PPO(
            policy="MlpPolicy",
            n_steps=5,
            env=None,
            _init_setup_model=False,
            device="cpu",
            seed=1)

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
    cycle_limit = 5000
    adaptation_limit = 50
    stagnation_limit = 5
    eval_frequency = 5
    eval_grp_size = 5
    logging = Log.C_LOG_WE
    visualize = True
    path = str(Path.home())

else:
    # 2.2 Parameters for internal unit test
    cycle_limit = 50
    adaptation_limit = 5
    stagnation_limit = 5
    eval_frequency = 2
    eval_grp_size = 1
    logging = Log.C_LOG_NOTHING
    visualize = False
    path = None

# 2.3 Create and run training object
training = RLTraining(
    p_scenario_cls=MyScenario,
    p_cycle_limit=cycle_limit,
    p_adaptation_limit=adaptation_limit,
    p_stagnation_limit=stagnation_limit,
    p_eval_frequency=eval_frequency,
    p_eval_grp_size=eval_grp_size,
    p_path=path,
    p_visualize=visualize,
    p_logging=logging)

training.run()