## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_env_004_train_agent_with_sb3_policy_on_double_pendulum_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-03-22  0.0.0     WB       Creation
## -- 2022-08-14  1.0.0     LSB      Training howto released with a lower value of torque
## -- 2022-09-09  1.0.1     SY       Refactoring and add DDPG algorithm as an option
## -- 2022-10-13  1.0.2     SY       Refactoring
## -- 2022-11-18  1.0.3     LSB      Refactoring for new plot style
## -- 2023-02-23  1.1.0     DA       Renamed
## -- 2023-03-02  1.1.1     LSB      Refactoring
## -- 2023-03-02  1.1.1     LSB      Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.1 (2023-03-02)

This module shows how to train double pendulum using on-policy and off-policy RL algorithms from SB3.

You will learn:

1. How to use MLPro's native Double Pendulum Environment for S7 variant.

2. How to create on-policy and off-policy objects for respective SP3 policies.

3. How to wrap the SB3 policies in MLPro.

4. How to setup and run RLTraining.
"""


import torch
from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.pool.envs.doublependulum import *
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from mlpro.wrappers.sb3 import WrPolicySB32MLPro
from mlpro.wrappers.openai_gym import WrEnvMLPro2GYM
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np



# 1 Implement your own RL scenario
class ScenarioDoublePendulum(RLScenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_visualize, p_logging):
        # 1.1 Setup environment
        self._env   = DoublePendulumS7(p_logging=p_logging, p_init_angles='random', p_max_torque=10,
                                        p_visualize=p_visualize, p_plot_level=DoublePendulumRoot.C_PLOT_DEPTH_ALL,
                                        p_reward_window=100)

        # 1.2 Select an algorithm by uncomment the opted algorithm
        # On-Policy RL Algorithm: A2C
        policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[dict(pi=[128, 128], vf=[128, 128])])
        policy_sb3 = A2C(
                    policy="MlpPolicy",
                    n_steps=150, 
                    env=None,
                    _init_setup_model=False,
                    policy_kwargs=policy_kwargs,
                    seed=1)
        
        # Off-Policy RL Algorithm: DDPG
        # action_space = WrEnvMLPro2GYM.recognize_space(self._env.get_action_space())
        # n_actions = action_space.shape[-1]
        # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        # policy_kwargs = dict(net_arch=dict(pi=[128, 128], qf=[128, 128]))
        # policy_sb3 = DDPG(
        #     policy="MlpPolicy",
        #     learning_rate=3e-4,
        #     buffer_size=10000,
        #     learning_starts=100,
        #     action_noise=action_noise,
        #     policy_kwargs=policy_kwargs,
        #     env=None,
        #     _init_setup_model=False,
        #     device="cpu")
            
        # 1.3 Wrapped the SB3 policy to MLPro compatible policy
        policy_wrapped = WrPolicySB32MLPro(
                p_sb3_policy=policy_sb3,
                p_cycle_limit=self._cycle_limit, 
                p_observation_space=self._env.get_state_space(),
                p_action_space=self._env.get_action_space(),
                p_ada=p_ada,
                p_logging=p_logging)

        # 1.4 Setup standard single-agent with the wrapped policy
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
    cycle_limit         = 100000
    adaptation_limit    = 0
    stagnation_limit    = 0
    eval_frequency      = 5
    eval_grp_size       = 3
    logging             = Log.C_LOG_WE
    visualize           = True
    path                = str(Path.home())
    plotting            = False
else:
    # 2.2 Parameters for unittest
    cycle_limit         = 0
    adaptation_limit    = 1
    stagnation_limit    = 0
    eval_frequency      = 5
    eval_grp_size       = 5
    logging             = Log.C_LOG_NOTHING
    visualize           = False
    path                = None
    plotting            = False
 

# 3 Train agent in scenario 
training        = RLTraining(
    p_scenario_cls=ScenarioDoublePendulum,
    p_cycle_limit=cycle_limit,
    p_cycles_per_epi_limit=150,
    p_adaptation_limit=adaptation_limit,
    p_stagnation_limit=stagnation_limit,
    p_eval_frequency=eval_frequency,
    p_eval_grp_size=eval_grp_size,
    p_path=path,
    p_visualize=visualize,
    p_logging=logging
)

training.run()
