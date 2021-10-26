import gym
from gym.wrappers.time_limit import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np

from mlpro.rl.wrappers import WrEnvGYM2MLPro
from mlpro.rl.wrappers import WrEnvMLPro2GYM

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, p_limit_episode, p_verbose=0):
        super(CustomCallback, self).__init__(p_verbose)
        self.episode_num = 0
        self.episode_limit = p_limit_episode
        self.total_cycle = 0
        self.cycles = 0
        self.plots = None
        self.continue_training = True
        self.rewards_cnt = []
        self.state = []
        self.action = []
        self.dones = []

    def _on_step(self) -> bool:
        self.total_cycle += 1
        self.cycles += 1
        self.state.append(self.locals.get("obs_tensor"))
        self.action.append(self.locals.get("actions"))
        self.dones.append(self.locals.get("dones"))
        if self.locals.get("infos")[0]:
            print(self.episode_num, self.total_cycle, self.locals.get("infos")[0]["episode"]["r"], self.cycles)
            self.rewards_cnt.append(self.locals.get("infos")[0]["episode"]["r"])
            self.episode_num += 1
            self.total_cycle = 0
            if self.episode_num >= self.episode_limit:
                return False
        
        return True

    def _on_training_end(self) -> None:
        self.plots = self.rewards_cnt

    def get_data_plot(self):
        return self.plots

    def get_state(self):
        return self.state

    def get_action(self):
        return self.action

    def get_dones(self):
        return self.dones

max_episode = 200
buffer_size = 1000000
policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                     net_arch=[dict(pi=[64, 64], vf=[64, 64])])

env = gym.make("CartPole-v1")
env2 = gym.make("CartPole-v1")
env3 = gym.make("CartPole-v1")
env.seed(1)
env2.seed(1)
env3.seed(1)

doublewrapped = WrEnvGYM2MLPro(env3, p_logging=False) 
doublewrapped = WrEnvMLPro2GYM(doublewrapped)
doublewrapped = TimeLimit(doublewrapped, max_episode_steps=100)

cus_callback = CustomCallback(p_limit_episode=max_episode)
model = PPO("MlpPolicy", env, n_steps=buffer_size, verbose=0, seed=1, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=1000000, callback=cus_callback)

cus_callback2 = CustomCallback(p_limit_episode=max_episode)
model2 = PPO("MlpPolicy", env2, n_steps=buffer_size, verbose=0, seed=1, policy_kwargs=policy_kwargs)
model2.learn(total_timesteps=1000000, callback=cus_callback2)

cus_callback3 = CustomCallback(p_limit_episode=max_episode)
model3 = PPO("MlpPolicy", doublewrapped, n_steps=buffer_size, verbose=0, seed=1, policy_kwargs=policy_kwargs)
model3.learn(total_timesteps=1000000, callback=cus_callback3)

state_native = cus_callback.get_state()
state_native2 = cus_callback2.get_state()
state_wrapped = cus_callback3.get_state()

action_native = cus_callback.get_action()
action_native2 = cus_callback2.get_action()
action_wrapped = cus_callback3.get_action()

done_native = cus_callback.get_dones()
done_native2 = cus_callback2.get_dones()
done_wrapped = cus_callback3.get_dones()

lengths = min(len(state_native), len(state_wrapped))

dataPlotState1 = []
dataPlotState2 = []
dataPlotAction1 = []
dataPlotAction2 = []
dataPlotDones1 = []
dataPlotDones2 = []

for x in range(lengths):
    # State
    dataPlotState1.append(torch.norm(state_native[x]-state_wrapped[x]).item())
    dataPlotState2.append(torch.norm(state_native[x]-state_native2[x]).item())

    # Action
    dataPlotAction1.append(np.linalg.norm(action_native[x]-action_wrapped[x]))
    dataPlotAction2.append(np.linalg.norm(action_native[x]-action_native2[x]))

    # Dones
    dataPlotDones1.append(done_native[x]^done_wrapped[x])
    dataPlotDones2.append(done_native[x]^done_native2[x])

# plt.plot(cus_callback.get_data_plot(), label="ENV NATIVE")
# plt.plot(cus_callback2.get_data_plot(), label="ENV2 NATIVE")
# plt.plot(cus_callback3.get_data_plot(), label="DOUBLE WRAPPED")
plt.plot(dataPlotState1, label="State Diff Native and Wrapped")
plt.plot(dataPlotState2, label="State Diff Native and Native")
plt.xlabel("Timestep")
plt.legend()
plt.show()

plt.clf()
plt.plot(dataPlotAction1, label="Action Diff Native and Wrapped")
plt.plot(dataPlotAction2, label="Action Diff Native and Native")
plt.xlabel("Timestep")
plt.legend()
plt.show()

plt.clf()
plt.plot(dataPlotDones1, label="Done Diff Native and Native")
plt.plot(dataPlotDones2, label="Done Diff Native and Wrapped")
plt.xlabel("Timestep")
plt.legend()
plt.show()
