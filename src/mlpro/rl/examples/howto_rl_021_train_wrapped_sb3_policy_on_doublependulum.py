## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : howto_rl_021_train_wrapped_sb3_policy_on_doublependulum.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-03-22  0.0.0     WB       Creation
## -------------------------------------------------------------------------------------------------


"""
Ver. 0.0.0 (2022-03-22)

This module shows how to use SB3 wrapper to train double pendulum. Currently under construction...
"""


import torch
from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.pool.envs.doublependulum import DoublePendulum
from stable_baselines3 import A2C
from mlpro.wrappers.sb3 import WrPolicySB32MLPro
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np



# 1 Implement your own RL scenario
class ScenarioDoublePendulum(RLScenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):
        # 1 Setup environment
        self._env   = DoublePendulum(p_logging=True, init_angles='up', max_torque=50)

        policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                     net_arch=[dict(pi=[128, 128], vf=[128, 128])])

        policy_sb3 = A2C(
                    policy="MlpPolicy",
                    n_steps=100, 
                    env=None,
                    _init_setup_model=False,
                    policy_kwargs=policy_kwargs,
                    seed=1)

        policy_wrapped = WrPolicySB32MLPro(
                p_sb3_policy=policy_sb3,
                p_cycle_limit=self._cycle_limit, 
                p_observation_space=self._env.get_state_space(),
                p_action_space=self._env.get_action_space(),
                p_ada=p_ada,
                p_logging=p_logging)

        # 2 Setup standard single-agent with own policy
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
    cycle_limit         = 0
    adaptation_limit    = 6000
    stagnation_limit    = 0
    eval_frequency      = 5
    eval_grp_size       = 5
    logging             = Log.C_LOG_WE
    visualize           = True
    path                = str(Path.home())
    plotting            = True
else:
    # 2.2 Parameters for demo mode
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



# 4 Create Plotting Class
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
                ax.set_ylim(minval-(abs(minval)*0.1), maxval+(abs(maxval)*0.1))
                ax.set_xlabel("Episode")
                ax.legend(label, bbox_to_anchor = (1,0.5), loc = "center left")
                self.plots[0].append(name)
                self.plots[1].append(ax)
                if self.showing:
                    plt.show()
                else:
                    plt.close(fig)


# 5 Plotting 1 MLpro    
data_printing   = {"Cycle":        [False],
                    "Day":          [False],
                    "Second":       [False],
                    "Microsecond":  [False],
                    "Smith":        [True,-1]}


mem = training.get_results().ds_rewards
mem_plot    = MyDataPlotting(mem, p_showing=plotting, p_printing=data_printing)
mem_plot.get_plots()
