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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2021-10-18)

This module shows how to train with SB3 Wrapper for On-Policy Algorithm
"""

import gym
import pandas as pd
from stable_baselines3 import A2C, PPO, DQN, DDPG, SAC
from stable_baselines3.common.callbacks import BaseCallback
from mlpro.rl.models import *
from mlpro.rl.wrappers import WrEnvGYM2MLPro
from mlpro.rl.wrappers import WrPolicySB32MLPro
from mlpro.rl.wrappers import WrEnvMLPro2GYM
from mlpro.rl.pool.envs.robotinhtm import RobotHTM

max_episode = 150
mva_window = 1
buffer_size = 500

# 1 Implement your own RL scenario
class MyScenario(Scenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):
        # 1 Setup environment
        # mlpro_env = RobotHTM(p_seed=1, p_logging=False)
        # gym_env     = gym.make('MountainCarContinuous-v0')
        gym_env     = gym.make('Acrobot-v1')
        # gym_env     = gym.make('LunarLanderContinuous-v2')
        # gym_env     = gym.make('CartPole-v1')
        gym_env.seed(2)
        # self._env   = mlpro_env
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
                    env=None,
                    n_steps=buffer_size,
                    _init_setup_model=False,
                    seed=2)

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
    p_cycle_limit=-1,           # get cycle limit from environment
    p_visualize=False,
    p_logging=False
)

# 3 Copy the SB3 Policy
sb3_pol = copy.deepcopy(myscenario._agent._policy.sb3.policy)

# 4 Instantiate training
training        = Training(
    p_scenario=myscenario,
    p_episode_limit=max_episode,
    p_collect_states=True,
    p_collect_actions=True,
    p_collect_rewards=True,
    p_collect_training=True,
    p_logging=True
)

# 5 Train
training.run()

# 6 Create Plotting Class
class MyDataPlotting(DataPlotting):
    def get_plots(self):
        """
        A function to plot data
        """
        for name in self.data.names:
            maxval  = 0
            minval  = 0
            if self.printing[name][0]:
                fig     = plt.figure(figsize=(7,7))
                raw   = []
                label   = []
                ax = fig.subplots(1,1)
                ax.set_title(name)
                ax.grid(True, which="both", axis="both")
                for fr_id in self.data.frame_id[name]:
                    raw.append(np.sum(self.data.get_values(name,fr_id)))
                    if self.printing[name][1] == -1:
                        maxval = max(raw)
                        minval = min(raw)
                    else:
                        maxval = self.printing[name][2]
                        minval = self.printing[name][1]
                    
                    label.append("%s"%fr_id)
                ax.plot(raw)
                ax.set_ylim(minval-(abs(minval)*0.1), maxval+(maxval*0.1))
                ax.set_xlabel("Episode")
                ax.legend(label, bbox_to_anchor = (1,0.5), loc = "center left")
                self.plots[0].append(name)
                self.plots[1].append(ax)
                if self.showing:
                    plt.show()
                else:
                    plt.close(fig)

# 7 Plotting 1 MLpro    
data_printing   = {"Cycle":        [False],
                    "Day":          [False],
                    "Second":       [False],
                    "Microsecond":  [False],
                    "Smith":        [True,-1]}


_,_,_,mem = training.get_data()
mem_plot    = MyDataPlotting(mem, p_showing=False, p_printing=data_printing)
mem_plot.get_plots()
wrapper_plot = mem_plot.plots

# 8 Create Callback for the SB3 Training
class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, p_limit_episode, p_verbose=0):
        super(CustomCallback, self).__init__(p_verbose)
        reward_space = Set()
        reward_space.add_dim(Dimension(0, "Native"))
        self.ds_rewards  = RLDataStoring(reward_space)
        self.episode_num = 0
        self.episode_limit = p_limit_episode
        self.total_cycle = 0
        self.cycles = 0
        self.plots = None

        self.continue_training = True
        self.rewards_cnt = []

    def _on_training_start(self) -> None:
        self.ds_rewards.add_episode(self.episode_num)

    def _on_step(self) -> bool:
        # Custom Env Without Cycle Limit
        # if self.continue_training:
        #     self.rewards_cnt.append(self.locals.get("rewards"))
        #     self.ds_rewards.memorize_row(self.total_cycle, timedelta(0,0,0), self.locals.get("rewards"))
        #     self.total_cycle += 1
        # else:
        #     return False

        # With Cycle Limit
        self.ds_rewards.memorize_row(self.total_cycle, timedelta(0,0,0), self.locals.get("rewards"))
        self.total_cycle += 1
        self.cycles += 1
        if self.locals.get("infos")[0]:
            print(self.episode_num, self.total_cycle, self.locals.get("infos")[0]["episode"]["r"], self.cycles)
            self.episode_num += 1
            self.total_cycle = 0
            if self.episode_num >= self.episode_limit:
                return False
            self.ds_rewards.add_episode(self.episode_num)
        
        return True

    # Custom Env Without Cycle Limit
    # def _on_rollout_end(self) -> None:
    #     print(self.episode_num, self.total_cycle, sum(self.rewards_cnt))
    #     self.episode_num += 1
    #     self.total_cycle = 0
    #     if self.episode_num >= self.episode_limit:
    #         self.continue_training = False
    #     else:
    #         self.rewards_cnt = []
    #         self.ds_rewards.add_episode(self.episode_num)

    def _on_training_end(self) -> None:
        data_printing   = {"Cycle":        [False],
                            "Day":          [False],
                            "Second":       [False],
                            "Microsecond":  [False],
                            "Native":        [True,-1]}
        mem_plot    = MyDataPlotting(self.ds_rewards, p_showing=False, p_printing=data_printing)
        mem_plot.get_plots()
        self.plots = mem_plot.plots

# 9 Run the SB3 Training
# mlpro_env = RobotHTM(p_seed=1, p_logging=False)
# gym_env = WrEnvMLPro2GYM(mlpro_env)
# gym_env     = gym.make('MountainCarContinuous-v0')
gym_env     = gym.make('Acrobot-v1')
# gym_env     = gym.make('LunarLanderContinuous-v2')
# gym_env     = gym.make('CartPole-v1')
gym_env.seed(2)
policy_sb3 = PPO(
                policy="MlpPolicy", 
                env=gym_env,
                n_steps=buffer_size,
                verbose=0,
                seed=2)

cus_callback = CustomCallback(p_limit_episode=max_episode)
policy_sb3.learn(total_timesteps=10000000, callback=cus_callback)
native_plot = cus_callback.plots

# 10 Difference Plot
native_ydata = native_plot[1][0].lines[0].get_ydata()
wrapper_ydata = wrapper_plot[1][0].lines[0].get_ydata()
smoothed_native = pd.Series.rolling(pd.Series(native_ydata), mva_window).mean()
smoothed_native = [elem for elem in smoothed_native]
smoothed_wrapper = pd.Series.rolling(pd.Series(wrapper_ydata), mva_window).mean()
smoothed_wrapper = [elem for elem in smoothed_wrapper]
plt.plot(smoothed_native, label="Native")
plt.plot(smoothed_wrapper, label="Wrapper")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.show()