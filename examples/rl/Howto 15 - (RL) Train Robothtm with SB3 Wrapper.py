## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : Train Robothtm with SB3 Wrapper
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-01  0.0.0     MRD      Creation
## -- 2021-12-01  1.0.0     MRD      First Release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-12-01)

This module shows how to use SB3 wrapper to train Robothtm
"""


from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.pool.envs.robotinhtm import RobotHTM
from stable_baselines3 import PPO
from mlpro.wrappers.sb3 import WrPolicySB32MLPro
from pathlib import Path

# 1 Implement your own RL scenario
class ScenarioRobotHTM(RLScenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):
        # 1 Setup environment
        self._env   = RobotHTM(p_logging=True)

        policy_sb3 = PPO(
                    policy="MlpPolicy",
                    n_steps=100, 
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

# 2 Create scenario and start training

if __name__ == "__main__":
    # 2.1 Parameters for demo mode
    logging     = Log.C_LOG_ALL
    visualize   = True
    path        = str(Path.home())
    plotting    = True
    timestep    = 300000
 
else:
    # 2.2 Parameters for internal unit test
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    path        = None
    plotting    = False
    timestep    = 200


# 3 Instantiate scenario
myscenario  = ScenarioRobotHTM(
    p_mode=Environment.C_MODE_SIM,
    p_ada=True,
    p_cycle_limit=100,
    p_visualize=visualize,
    p_logging=logging
)

# 4 Train agent in scenario 
now             = datetime.now()

training        = RLTraining(
    p_scenario=myscenario,
    p_cycle_limit=timestep,
    p_max_cycles_per_episode=100,
    p_max_stagnations=0,
    p_collect_states=True,
    p_collect_actions=True,
    p_collect_rewards=True,
    p_collect_training=True,
    p_path=path,
    p_logging=logging
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
mem_plot    = MyDataPlotting(mem, p_showing=plotting, p_printing=data_printing)
mem_plot.get_plots()
