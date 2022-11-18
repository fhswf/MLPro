## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : experiments
## -- Module  : DP_001_A2C.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-11-18  0.0.0     SY       Creation
## -- 2022-11-18  1.0.0     SY       Release first version 
## -------------------------------------------------------------------------------------------------


"""
Ver. 1.0.0 (2022-11-18)

This module shows how to train double pendulum using A2C from SB3.
"""


import torch
from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.pool.envs.doublependulum import *
from stable_baselines3 import A2C
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
        self._env   = DoublePendulumS7(p_logging=True, p_init_angles='down', p_max_torque=10, p_visualize=True,
        p_plot_level=
        DoublePendulumRoot.C_PLOT_DEPTH_ALL)

        # 1.2 Select an algorithm
        # On-Policy RL Algorithm: A2C
        
        # Parameters, refer to https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html
        actor_size = 128
        critic_size = 128
        learning_rate = 7e-4
        n_steps = 5
        gamma = 0.99
        
        policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[dict(pi=[actor_size, actor_size], vf=[critic_size, critic_size])])
        policy_sb3 = A2C(
                    policy="MlpPolicy",
                    env=None,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    gamma=gamma, 
                    _init_setup_model=False,
                    policy_kwargs=policy_kwargs,
                    seed=1)
            
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
