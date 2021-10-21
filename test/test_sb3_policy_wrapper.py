## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : test_policy_wrapper
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-11  1.0.0     MRD      Creation
## -- 2021-09-21  1.0.0     MRD      Release First Version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-09-21)

Unit test classes for environment.
"""


import pytest
import gym
from mlpro.rl.models import *
from mlpro.rl.wrappers import WrEnvGYM2MLPro
from mlpro.rl.wrappers import WrPolicySB32MLPro
from stable_baselines3 import A2C, PPO, DQN, DDPG, SAC

## -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("env_cls", [A2C, PPO, DQN, DDPG, SAC])
def test_sb3_policy_wrapper(env_cls):
    class MyScenario(Scenario):

        C_NAME      = 'Matrix'

        def _setup(self, p_mode, p_ada, p_logging):
            # 1 Setup environment
            gym_env     = gym.make('MountainCarContinuous-v0')
            self._env   = WrEnvGYM2MLPro(gym_env, p_logging=False)
            

            param = {"policy": "MlpPolicy", "env": None, "_init_setup_model": False}

            if env_cls == A2C:
                param = {**param, **{"use_rms_prop": False}}

            if env_cls == DQN:
                print("Here")
                gym_env     = gym.make('CartPole-v1')
                self._env   = WrEnvGYM2MLPro(gym_env, p_logging=False)

            policy_sb3 = env_cls(
                        policy="MlpPolicy", 
                        env=None,
                        _init_setup_model=False)


            # 3 Wrap the policy
            policy_wrapped = WrPolicySB32MLPro(
                    p_sb3_policy=policy_sb3, 
                    p_observation_space=self._env.get_state_space(),
                    p_action_space=self._env.get_action_space(),
                    p_buffer_size=3,
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
        p_cycle_limit=50,
        p_visualize=False,
        p_logging=False
    )

    # 3 Instantiate training
    training        = Training(
        p_scenario=myscenario,
        p_episode_limit=6,
        p_cycle_limit=50,
        p_collect_states=True,
        p_collect_actions=True,
        p_collect_rewards=True,
        p_collect_training=True,
        p_logging=True
    )

    # 4 Train
    training.run()
