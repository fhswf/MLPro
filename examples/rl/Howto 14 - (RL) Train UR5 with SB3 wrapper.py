## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : Howto 07 - Train UR5 Robot environment with A2C Algorithm
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-23  0.0.0     WB       Creation
## -- 2021-09-23  1.0.0     WB       Released first version
## -- 2021 09-26  1.0.1     MRD      Change the import module due to the change of the pool
## --                                folder structer
## -- 2021-10-18  1.0.2     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2021-10-18)

This module shows how to implement A2C on the UR5 Robot Environment
"""


from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.pool.envs.ur5jointcontrol import UR5JointControl
from stable_baselines3 import PPO
from mlpro.rl.wrappers import WrPolicySB32MLPro
import random
from pathlib import Path

# 1 Make Sure training_env branch of ur_control is sourced:
    # request access to the ur_control project

# 2 Implement your own RL scenario
class ScenarioUR5A2C(RLScenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):
        # 1 Setup environment
        self._env   = UR5JointControl(p_logging=True)

        policy_sb3 = PPO(
                    policy="MlpPolicy",
                    n_steps=20, 
                    env=None,
                    _init_setup_model=False)

        policy_wrapped = WrPolicySB32MLPro(
                p_sb3_policy=policy_sb3, 
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




# 3 Instantiate scenario
myscenario  = ScenarioUR5A2C(
    p_mode=Environment.C_MODE_SIM,
    p_ada=True,
    p_cycle_limit=20,
    p_visualize=True,
    p_logging=False
)




# 4 Train agent in scenario 
now             = datetime.now()

training        = RLTraining(
    p_scenario=myscenario,
    p_cycle_limit=100000,
    p_max_stagnations=0,
    p_collect_states=True,
    p_collect_actions=True,
    p_collect_rewards=True,
    p_collect_training=True,
    p_logging=True
)

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
                ax.set_ylim(minval-(abs(minval)*0.1), maxval+(abs(maxval)*0.1))
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


mem = training.get_results().ds_rewards
mem_plot    = MyDataPlotting(mem, p_showing=False, p_printing=data_printing)
mem_plot.get_plots()