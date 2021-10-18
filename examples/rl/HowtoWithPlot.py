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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2021-10-08)

This module shows how to train with SB3 Wrapper for On-Policy Algorithm
"""

import gym
from stable_baselines3 import A2C, PPO, DQN, DDPG, SAC
from mlpro.rl.models import *
from mlpro.rl.wrappers import WrEnvGYM2MLPro
from mlpro.rl.wrappers import WrPolicySB32MLPro
from mlpro.rl.pool.envs.robotinhtm import RobotHTM
from collections import deque
import pandas as pd

# 1 Implement your own RL scenario
class MyScenario(Scenario):

    C_NAME      = 'Matrix'

    def __init__(self, p_mode=..., p_ada=True, p_cycle_len: timedelta = None, p_cycle_limit=0, p_visualize=True, p_logging=True):
        super().__init__(p_mode=p_mode, p_ada=p_ada, p_cycle_len=p_cycle_len, p_cycle_limit=p_cycle_limit, p_visualize=p_visualize, p_logging=p_logging)
        self._reward_training = []

    def _setup(self, p_mode, p_ada, p_logging):
        # 1 Setup environment
        # self._env   = RobotHTM(p_logging=False)
        gym_env     = gym.make('MountainCarContinuous-v0')
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
        # policy_sb3 = PPO(
        #             policy="MlpPolicy", 
        #             env=None,
        #             _init_setup_model=False)

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
        policy_sb3 = SAC(
                    policy="MlpPolicy", 
                    env=None,
                    _init_setup_model=False)

        # 3 Wrap the policy
        policy_wrapped = WrPolicySB32MLPro(
                p_sb3_policy=policy_sb3, 
                p_state_space=self._env.get_state_space(),
                p_action_space=self._env.get_action_space(),
                p_buffer_size=1000000,
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

    def run_cycle(self, p_cycle_id, p_ds_states:RLDataStoring=None, p_ds_actions:RLDataStoring=None, 
                p_ds_rewards:RLDataStoring=None):
        """
        Processes a single control cycle with optional data logging.

        Parameters:
            p_cycle_id          Cycle id
            p_ds_states         Optional external data storing object that collects environment state data
            p_ds_actions        Optional external data storing object that collects agent action data
            p_ds_rewards        Optional external data storing object that collects environment reeward data
        """

        # 0 Cycle intro
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Start of cycle', str(p_cycle_id))


        # 1 Environment: get and log current state
        state   = self._env.get_state()
        if p_ds_states is not None:
            p_ds_states.memorize_row(p_cycle_id, self._timer.get_time(), state.get_values())


        # 2 Agent: compute and log next action
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Agent computes action...')
        action  = self._agent.compute_action(state)
        ts      = self._timer.get_time()
        action.set_tstamp(ts)
        if p_ds_actions is not None:
            p_ds_actions.memorize_row(p_cycle_id, ts, action.get_sorted_values())


        # 3 Environment: process agent's action
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Env processes action...')
        self._env.process_action(action)
        self._timer.add_time(self._env.get_latency())     # in virtual mode only...
        self._env.get_state().set_tstamp(self._timer.get_time())


        # 4 Environment: compute and log reward
        reward  = self._env.compute_reward()
        ts      = self._timer.get_time()
        reward.set_tstamp(ts)
        if p_ds_rewards is not None:
            if ( reward.get_type() == Reward.C_TYPE_OVERALL ) or ( reward.get_type() == Reward.C_TYPE_EVERY_AGENT ):
                reward_values = np.zeros(p_ds_rewards.get_space().get_num_dim())

                for i, agent_id in enumerate(p_ds_rewards.get_space().get_dim_ids()): 
                    reward_values[i] = reward.get_agent_reward(agent_id)
                
                p_ds_rewards.memorize_row(p_cycle_id, ts, reward_values)


        # 5 Agent: adapt policy
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Agent adapts policy...')
        self._agent.adapt(self._env.get_state(), reward)
        self._reward_training.append(reward.get_overall_reward())

        # 6 Optional visualization
        if self._visualize:
            self._env.update_plot()
            self._agent.update_plot()


        # 7 Wait for next cycle (virtual mode only)
        if ( self._timer.finish_lap() == False ) and ( self.cycle_len is not None ):
            self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Process timed out !!!')


        # 8 Cycle outro
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': End of cycle', str(p_cycle_id), '\n')

# 2 Instantiate scenario
myscenario  = MyScenario(
    p_mode=Environment.C_MODE_SIM,
    p_ada=True,
    p_cycle_limit=-1,           # get cycle limit from environment
    p_visualize=False,
    p_logging=False
)

class MyTraining(Training):
    def __init__(self, p_scenario: Scenario, p_episode_limit=50, p_cycle_limit=0, p_collect_states=True, p_collect_actions=True, p_collect_rewards=True, p_collect_training=True, p_logging=True):
        super().__init__(p_scenario, p_episode_limit=p_episode_limit, p_cycle_limit=p_cycle_limit, p_collect_states=p_collect_states, p_collect_actions=p_collect_actions, p_collect_rewards=p_collect_rewards, p_collect_training=p_collect_training, p_logging=p_logging)
        self._training_plot = []
        self._reward_buffer = deque(maxlen=100)

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
        if self._env.get_done() or self._env.get_broken() or ( self._cycle_id == (self._cycle_limit-1) ):
            # 3.1 Episode finished
            self.log(self.C_LOG_TYPE_I, '--------------------------------------')
            self.log(self.C_LOG_TYPE_I, '-- Episode', self._episode_id, 'finished after', self._cycle_id + 1, 'cycles')
            self.log(self.C_LOG_TYPE_I, '--------------------------------------\n\n')

            # 3.1.1 Update global training data storage
            if self._ds_training is not None:
                if self._env.get_done()==True:
                    done_num = 1
                else:
                    done_num = 0

                if self._env.get_broken()==True:
                    broken_num = 1
                else:
                    broken_num = 0

                self._ds_training.memorize(RLDataStoring.C_VAR_NUM_CYLCLES, self._episode_id, self._cycle_id + 1)
                self._ds_training.memorize(RLDataStoring.C_VAR_ENV_DONE, self._episode_id, done_num)
                self._ds_training.memorize(RLDataStoring.C_VAR_ENV_BROKEN, self._episode_id, broken_num)
 
            self._reward_buffer.extend([np.sum(np.array(self._scenario._reward_training))])
            self._scenario._reward_training = []
            # 3.1.2 Prepare next episode
            self._episode_id   += 1
            self._cycle_id      = 0

        else:
            # 3.2 Prepare next cycle
            self._cycle_id     += 1

    def run_episode(self):
        """
        Runs/finishes current training episode.
        """

        current_episode_id = self._episode_id
        while self._episode_id == current_episode_id: self.run_cycle()
        print(np.mean(self._reward_buffer))
        self._training_plot.append(np.mean(self._reward_buffer))

    def run(self):
        """
        Runs/finishes entire training.
        """

        while self._episode_id < self._episode_limit: self.run_episode()
        smoothed = pd.Series.rolling(pd.Series(self._training_plot), 10).mean()
        smoothed = [elem for elem in smoothed]
        plt.plot(self._training_plot)
        plt.plot(smoothed)
        plt.ylabel('some numbers')
        plt.show()

# 3 Instantiate training
training        = MyTraining(
    p_scenario=myscenario,
    p_episode_limit=100,
    p_collect_states=True,
    p_collect_actions=True,
    p_collect_rewards=True,
    p_collect_training=True,
    p_logging=True
)

# 4 Train
training.run()