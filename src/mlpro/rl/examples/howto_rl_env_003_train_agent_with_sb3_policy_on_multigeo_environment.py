## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_env_003_train_agent_with_sb3_policy_on_multigeo_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-19  0.0.0     MRD      Creation
## -- 2021-12-19  1.0.0     MRD      Initial Release
## -- 2021-12-23  1.0.1     DA       Minor fix 
## -- 2022-10-13  1.0.2     SY       Refactoring 
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2022-10-13)

This module shows how to use SB3 wrapper to train Multi Geometry Robot.

You will learn:
    
1) How to set up a scenario for Multi Geometry Robot and also with SB3 wrapper

2) How to run the scenario and train the agent
    
3) How to plot from the generated results
    
"""

from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.pool.envs.multigeorobot import MultiGeo
from stable_baselines3 import PPO
from mlpro.wrappers.sb3 import WrPolicySB32MLPro
from pathlib import Path


# 1 Implement your own RL scenario
class ScenarioMultiGeoPPO(RLScenario):
    C_NAME = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):
        # 1.1 Setup environment
        self._env = MultiGeo(p_logging=p_logging)

        policy_sb3 = PPO(
            policy="MlpPolicy",
            n_steps=20,
            env=None,
            _init_setup_model=False,
            device="cpu",
            seed=1)

        policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=policy_sb3,
            p_cycle_limit=self._cycle_limit,
            p_observation_space=self._env.get_state_space(),
            p_action_space=self._env.get_action_space(),
            p_ada=p_ada,
            p_logging=p_logging)

        # 1.2 Setup standard single-agent with own policy
        return Agent(
            p_policy=policy_wrapped,
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_logging=p_logging
        )


# 2 Train agent in scenario
if __name__ == "__main__":
    # 2.1 Parameters for demo mode
    cycle_limit = 1000
    logging = Log.C_LOG_WE
    visualize = True
    path = str(Path.home())
    plotting = True

else:
    # 2.2 Parameters for internal unit test
    cycle_limit = 50
    logging = Log.C_LOG_NOTHING
    visualize = False
    path = None
    plotting = False

training = RLTraining(
    p_scenario_cls=ScenarioMultiGeoPPO,
    p_cycle_limit=cycle_limit,
    p_cycles_per_epi_limit=-1,
    p_collect_states=True,
    p_collect_actions=True,
    p_collect_rewards=True,
    p_collect_training=True,
    p_visualize=visualize,
    p_path=path,
    p_logging=plotting)

training.run()


# 3 Create Plotting Class
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


# 4 Plotting MLpro
if __name__ == "__main__": 
    data_printing = {"Cycle": [False],
                     "Day": [False],
                     "Second": [False],
                     "Microsecond": [False],
                     "Smith": [True, -1]}
    
    mem = training.get_results().ds_rewards
    mem_plot = MyDataPlotting(mem, p_showing=True, p_printing=data_printing)
    mem_plot.get_plots()
