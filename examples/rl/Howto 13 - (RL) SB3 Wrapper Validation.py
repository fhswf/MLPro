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
from stable_baselines3 import A2C, PPO, DQN, DDPG, SAC
from stable_baselines3.common.callbacks import BaseCallback
from mlpro.rl.models import *
from mlpro.rl.wrappers import WrEnvGYM2MLPro
from mlpro.rl.wrappers import WrPolicySB32MLPro


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
        # policy_sb3 = A2C(
        #             policy="MlpPolicy", 
        #             env=None,
        #             use_rms_prop=False, 
        #             _init_setup_model=False)

        # PPO
        policy_sb3 = PPO(
                    policy="MlpPolicy", 
                    env=None,
                    _init_setup_model=False)

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
    p_cycle_limit=-1,           # get cycle limit from environment
    p_visualize=False,
    p_logging=False
)

# 3 Copy the SB3 Policy
sb3_pol = copy.deepcopy(myscenario._agent._policy.sb3.policy)

# 4 Instantiate training
training        = Training(
    p_scenario=myscenario,
    p_episode_limit=200,
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
                plt.title(name)
                plt.grid(True, which="both", axis="both")
                for fr_id in self.data.frame_id[name]:
                    raw.append(np.sum(self.data.get_values(name,fr_id)))
                    if self.printing[name][1] == -1:
                        maxval = max(raw)
                        minval = min(raw)
                    else:
                        maxval = self.printing[name][2]
                        minval = self.printing[name][1]
                    
                    label.append("%s"%fr_id)
                plt.plot(self.moving_mean(raw, self.window))
                plt.ylim(minval-(abs(minval)*0.1), maxval+(maxval*0.1))
                plt.xlabel("Episode")
                plt.legend(label, bbox_to_anchor = (1,0.5), loc = "center left")
                self.plots[0].append(name)
                self.plots[1].append(fig)
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
mem_plot    = MyDataPlotting(mem, p_window=10, p_showing=True, p_printing=data_printing)
mem_plot.get_plots()

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

    def _on_training_start(self) -> None:
        self.ds_rewards.add_episode(self.episode_num)

    def _on_step(self) -> bool:
        if self.episode_num > self.episode_limit:
            return False

        self.ds_rewards.memorize_row(self.total_cycle, timedelta(0,0,0), self.locals.get("rewards"))
        self.total_cycle += 1
        if self.locals.get("infos")[0]:
            print(self.episode_num, self.total_cycle, self.locals.get("infos")[0]["episode"]["r"])
            self.episode_num += 1
            self.total_cycle = 0
            self.ds_rewards.add_episode(self.episode_num)
        
        return True

    def _on_training_end(self) -> None:
        data_printing   = {"Cycle":        [False],
                            "Day":          [False],
                            "Second":       [False],
                            "Microsecond":  [False],
                            "Native":        [True,-1]}
        mem_plot    = MyDataPlotting(self.ds_rewards, p_window=10, p_showing=True, p_printing=data_printing)
        mem_plot.get_plots()

# 9 Run the SB3 Training
env     = gym.make('CartPole-v1')
policy_sb3 = PPO(
                policy="MlpPolicy", 
                env=env,
                n_steps=500,
                verbose=0)
policy_sb3.policy = sb3_pol
policy_sb3.learn(total_timesteps=1000000, callback=CustomCallback(p_limit_episode=200))