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
from mlpro.rl.wrappers import WrEnvGym
from mlpro.rl.wrappers import WrPolicySB3
from mlpro.rl.pool.envs import RobotHTM
from collections import deque

# 1 Implement your own RL scenario
class MyScenario(Scenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):
        # 1 Setup environment
        # self._env   = RobotHTM(p_logging=False)
        gym_env     = gym.make('CartPole-v1')
        self._env   = WrEnvGym(gym_env, p_logging=False) 

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
        policy_wrapped = WrPolicySB3(
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


# 3 Implement Training Class for On-Policy
class OnPolicyTraining(Training):
    C_NAME      = 'On Policy Training'

    def run_cycle(self):
        """
        Runs next training cycle.
        """

        # 1 Begin of new episode? Reset agent and environment 
        if self._cycle_id == 0:
            self.log(self.C_LOG_TYPE_I, '--------------------------------------')
            self.log(self.C_LOG_TYPE_I, '-- Episode', self._episode_id, 'started...')
            self.log(self.C_LOG_TYPE_I, '--------------------------------------\n')
            self._scenario.reset()
 
            # 1.1 Init frame for next episode in data storage objects
            if self._ds_training is not None: self._ds_training.add_episode(self._episode_id)
            if self._ds_states is not None: self._ds_states.add_episode(self._episode_id)
            if self._ds_actions is not None: self._ds_actions.add_episode(self._episode_id)
            if self._ds_rewards is not None: self._ds_rewards.add_episode(self._episode_id)


        # 2 Run a cycle
        self._scenario.run_cycle(self._cycle_id, p_ds_states=self._ds_states, p_ds_actions=self._ds_actions, p_ds_rewards=self._ds_rewards)


        # 3 Update training counters
        if self._agent._policy._buffer.full or self._env.broken or ( self._cycle_id == (self._cycle_limit-1) ):
            # 3.1 Episode finished
            self.log(self.C_LOG_TYPE_I, '--------------------------------------')
            self.log(self.C_LOG_TYPE_I, '-- Episode', self._episode_id, 'finished after', self._cycle_id + 1, 'cycles')
            self.log(self.C_LOG_TYPE_I, '--------------------------------------\n\n')

            # 3.1.1 Update global training data storage
            if self._ds_training is not None:
                if self._env.done==True:
                    done_num = 1
                else:
                    done_num = 0

                if self._env.broken==True:
                    broken_num = 1
                else:
                    broken_num = 0

                self._ds_training.memorize(RLDataStoring.C_VAR_NUM_CYLCLES, self._episode_id, self._cycle_id + 1)
                self._ds_training.memorize(RLDataStoring.C_VAR_ENV_DONE, self._episode_id, done_num)
                self._ds_training.memorize(RLDataStoring.C_VAR_ENV_BROKEN, self._episode_id, broken_num)

            self._scenario.reset()
            # 3.1.2 Prepare next episode
            self._episode_id   += 1
            self._cycle_id      = 0
        elif self._env.done:
            self._scenario.reset()
            self._cycle_id     += 1
        else:
            # 3.2 Prepare next cycle
            self._cycle_id     += 1

# 4 Instantiate training
training        = OnPolicyTraining(
    p_scenario=myscenario,
    p_episode_limit=2000,
    p_cycle_limit=500,
    p_collect_states=True,
    p_collect_actions=True,
    p_collect_rewards=True,
    p_collect_training=True,
    p_logging=True
)

# 5 Train
training.run()