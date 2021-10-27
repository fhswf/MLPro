## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : Howto 13 - Comparison Native and Wrapper SB3 Policy
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-10-27  0.0.0     MRD      Creation
## -- 2021-10-27  1.0.0     MRD      Released first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-10-27)

This module shows how to train with SB3 Wrapper for On-Policy Algorithm
"""

import gym
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from mlpro.rl.models import *
from mlpro.rl.wrappers import WrEnvGYM2MLPro
from mlpro.rl.wrappers import WrPolicySB32MLPro

# 1 Parameter
max_episode = 400
mva_window = 1
buffer_size = 100
policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                     net_arch=[dict(pi=[10, 10], vf=[10, 10])])

# 2 Implement your own RL scenario
class MyScenario(Scenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):
        # 1 Setup environment
        gym_env     = gym.make('CartPole-v1')
        gym_env.seed(1)
        # self._env   = mlpro_env
        self._env   = WrEnvGYM2MLPro(gym_env, p_logging=False) 

        # 2 Instatiate Policy From SB3
        # env is set to None, it will be set up later inside the wrapper
        # _init_setup_model is set to False, the _setup_model() will be called inside
        # the wrapper manually

        # PPO
        policy_sb3 = PPO(
                    policy="MlpPolicy", 
                    env=None,
                    n_steps=buffer_size,
                    _init_setup_model=False,
                    policy_kwargs=policy_kwargs,
                    seed=1)

        # 3 Wrap the policy
        self.policy_wrapped = WrPolicySB32MLPro(
                p_sb3_policy=policy_sb3, 
                p_observation_space=self._env.get_state_space(),
                p_action_space=self._env.get_action_space(),
                p_ada=p_ada,
                p_logging=p_logging)
        
        # 4 Setup standard single-agent with own policy
        self._agent = Agent(
            p_policy=self.policy_wrapped,   
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_logging=p_logging
        )

# 3 Instantiate scenario
myscenario  = MyScenario(
    p_mode=Environment.C_MODE_SIM,
    p_ada=True,
    p_cycle_limit=-1,           # get cycle limit from environment
    p_visualize=False,
    p_logging=False
)

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

# 5 Train SB3 Wrapper
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
class CustomCallback(BaseCallback, Log):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    C_TYPE                  = 'Wrapper'
    C_NAME                  = 'SB3 Policy'

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
        self.log(self.C_LOG_TYPE_I, '--------------------------------------')
        self.log(self.C_LOG_TYPE_I, '-- Episode', self.episode_num, 'started...')
        self.log(self.C_LOG_TYPE_I, '--------------------------------------\n')
        self.ds_rewards.add_episode(self.episode_num)

    def _on_step(self) -> bool:
        # With Cycle Limit
        self.ds_rewards.memorize_row(self.total_cycle, timedelta(0,0,0), self.locals.get("rewards"))
        self.total_cycle += 1
        self.cycles += 1
        if self.locals.get("infos")[0]:
            self.log(self.C_LOG_TYPE_I, '--------------------------------------')
            self.log(self.C_LOG_TYPE_I, '-- Episode', self.episode_num, 'finished after', self.total_cycle + 1, 'cycles')
            self.log(self.C_LOG_TYPE_I, '--------------------------------------\n\n')
            self.episode_num += 1
            self.total_cycle = 0
            if self.episode_num >= self.episode_limit:
                return False
            self.ds_rewards.add_episode(self.episode_num)
            self.log(self.C_LOG_TYPE_I, '--------------------------------------')
            self.log(self.C_LOG_TYPE_I, '-- Episode', self.episode_num, 'started...')
            self.log(self.C_LOG_TYPE_I, '--------------------------------------\n')
        
        return True

    def _on_training_end(self) -> None:
        data_printing   = {"Cycle":        [False],
                            "Day":          [False],
                            "Second":       [False],
                            "Microsecond":  [False],
                            "Native":        [True,-1]}
        mem_plot    = MyDataPlotting(self.ds_rewards, p_showing=False, p_printing=data_printing)
        mem_plot.get_plots()
        self.plots = mem_plot.plots

# 9 Run the SB3 Training Native
gym_env     = gym.make('CartPole-v1')
gym_env.seed(1)
policy_sb3 = PPO(
                policy="MlpPolicy", 
                env=gym_env,
                n_steps=buffer_size,
                verbose=0,
                policy_kwargs=policy_kwargs,
                seed=1)

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