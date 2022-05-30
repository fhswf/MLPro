## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto_rl_013_model_based_reinforcement_learning.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-17  0.0.0     MRD       Creation
## -- 2021-12-17  1.0.0     MRD       Released first version
## -- 2022-01-01  1.0.1     MRD       Refactoring due to new model implementation
## -- 2022-05-20  1.0.2     MRD       Add HTMEnvModel
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2022-05-20)

This module demonstrates model-based reinforcement learning (MBRL).
"""


import torch

from mlpro.bf.ml import *
from mlpro.rl.models import *
from mlpro.rl.pool.envs.robotinhtm import RobotHTM
from stable_baselines3 import PPO
from mlpro.wrappers.sb3 import WrPolicySB32MLPro
from mlpro.rl.pool.envmodels.mlp_robotinhtm import MLPEnvModel
from mlpro.rl.pool.envmodels.htm_robotinhtm import HTMEnvModel

from pathlib import Path



class ActualTraining(RLTraining):
    C_NAME = "Actual"



# Implement RL Scenario for the actual environment to train the environment model
class ScenarioRobotHTMActual(RLScenario):
    C_NAME = "Matrix1"

    def _setup(self, p_mode, p_ada, p_logging):
        # 1 Setup environment
        self._env = RobotHTM(p_logging=True)

        policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                             net_arch=[dict(pi=[128, 128], vf=[128, 128])])

        policy_sb3 = PPO(
            policy="MlpPolicy",
            n_steps=100,
            env=None,
            _init_setup_model=False,
            policy_kwargs=policy_kwargs,
            device="cpu"
        )

        policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=policy_sb3,
            p_cycle_limit=self._cycle_limit,
            p_observation_space=self._env.get_state_space(),
            p_action_space=self._env.get_action_space(),
            p_ada=p_ada,
            p_logging=p_logging,
        )

        mb_training_param = dict(p_cycle_limit=100,
                                 p_cycles_per_epi_limit=100,
                                 p_max_stagnations=0,
                                 p_collect_states=False,
                                 p_collect_actions=False,
                                 p_collect_rewards=False,
                                 p_collect_training=False)

        # 2 Setup standard single-agent with own policy
        return Agent(
            p_policy=policy_wrapped,
            p_envmodel=HTMEnvModel(),
            p_em_mat_thsld=0.5,
            p_name="Smith1",
            p_ada=p_ada,
            p_logging=p_logging,
            **mb_training_param
        )


# 3 Train agent in scenario
if __name__ == "__main__":
    # 3.1 Parameters for demo mode
    cycle_limit = 300000
    logging     = Log.C_LOG_ALL
    visualize   = True
    path        = str(Path.home())
    plotting    = True
else:
    # 3.2 Parameters for internal unit test
    cycle_limit = 100
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    path        = None
    plotting    = False

training = ActualTraining(
    p_scenario_cls=ScenarioRobotHTMActual,
    p_cycle_limit=cycle_limit,
    p_cycles_per_epi_limit=100,
    p_collect_states=True,
    p_collect_actions=True,
    p_collect_rewards=True,
    p_collect_training=True,
    p_path=path,
    p_logging=logging,
)

training.run()


# 4 Create Plotting Class
class MyDataPlotting(DataPlotting):
    def get_plots(self):
        """
        A function to plot data
        """
        for name in self.data.names:
            maxval = 0
            minval = 0
            if self.printing[name][0]:
                fig = plt.figure(figsize=(7, 7))
                raw = []
                label = []
                ax = fig.subplots(1, 1)
                ax.set_title(name)
                ax.grid(True, which="both", axis="both")
                for fr_id in self.data.frame_id[name]:
                    raw.append(np.sum(self.data.get_values(name, fr_id)))
                    if self.printing[name][1] == -1:
                        maxval = max(raw)
                        minval = min(raw)
                    else:
                        maxval = self.printing[name][2]
                        minval = self.printing[name][1]

                    label.append("%s" % fr_id)
                ax.plot(raw)
                ax.set_ylim(minval - (abs(minval) * 0.1), maxval + (abs(maxval) * 0.1))
                ax.set_xlabel("Episode")
                ax.legend(label, bbox_to_anchor=(1, 0.5), loc="center left")
                self.plots[0].append(name)
                self.plots[1].append(ax)
                if self.showing:
                    plt.show()
                else:
                    plt.close(fig)


# 5 Plotting 1 MLpro
data_printing = {
    "Cycle": [False],
    "Day": [False],
    "Second": [False],
    "Microsecond": [False],
    "Smith1": [True, -1],
}

mem = training.get_results().ds_rewards
mem_plot = MyDataPlotting(mem, p_showing=plotting, p_printing=data_printing)
mem_plot.get_plots()
