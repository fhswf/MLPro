## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_mb_002_grid_world_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-09-19  0.0.0     SY       Creation
## -- 2022-10-06  1.0.0     SY       Release first version
## -- 2022-10-07  1.0.1     SY       Add plotting
## -- 2022-10-13  1.0.2     SY       Refactoring 
## -- 2022-11-07  1.1.0     DA       Refactoring 
## -- 2023-02-02  1.2.0     DA       Refactoring
## -- 2023-02-04  1.2.1     SY       Add multiprocessing functionality and refactoring
## -- 2023-02-10  1.2.2     SY       Switch multiprocessing to threading
## -- 2023-03-07  2.0.0     SY       Update due to MLPro-SL
## -- 2023-03-08  2.0.1     SY       Refactoring
## -- 2023-03-10  2.0.2     SY       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.0.2 (2023-03-10)

This module shows how to incorporate MPC in Model-Based RL on Grid World problem as well as using
PyTorch-based MLP network from MLPro-SL's pool of objects.

You will learn:
    
1) How to set up an own agent on Grid World problem
    
2) How to set up model-based RL (MBRL) training using network from MLPro-SL's pool
    
3) How to incorporate MPC into MBRL training

4) How to use multiprocessing on MPC
    
"""


import torch
import numpy as np
from mlpro.bf.plot import DataPlotting
from mlpro.bf.math import *
from mlpro.rl import *
from mlpro.rl.models import *
from mlpro.rl.pool.envs.gridworld import *
from mlpro.rl.pool.policies.randomgenerator import RandomGenerator
from mlpro.sl.pool.afct.fnn.pytorch.mlp import PyTorchMLP
from pathlib import Path
from mlpro.rl.pool.actionplanner.mpc import MPC
from mlpro.rl.pool.envs.gridworld import *
import mlpro.bf.mt as mt
              

            
                
                
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# 1 Implement the random RL scenario
class ScenarioGridWorld(RLScenario):

    C_NAME      = 'Grid World with Random Actions'


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        # 1.1 Setup environment
        self._env   = GridWorld(p_logging=p_logging,
                                p_action_type=GridWorld.C_ACTION_TYPE_DISC_2D,
                                p_max_step=100)


        # 1.2 Setup and return random action agent
        policy_random = RandomGenerator(p_observation_space=self._env.get_state_space(), 
                                        p_action_space=self._env.get_action_space(),
                                        p_buffer_size=1,
                                        p_ada=1,
                                        p_logging=p_logging)
        
        # Setup Adaptive Function
        afct_strans = AFctSTrans(
            PyTorchMLP,
            p_state_space=self._env._state_space,
            p_action_space=self._env._action_space,
            p_threshold=0.1,
            p_buffer_size=100,
            p_ada=p_ada,
            p_logging=p_logging,
            p_update_rate=1,
            p_num_hidden_layers=3,
            p_hidden_size=128,
            p_activation_fct=torch.nn.ReLU,
            p_output_activation_fct=torch.nn.ReLU,
            p_optimizer=torch.optim.Adam,
            p_loss_fct=torch.nn.MSELoss,
            p_learning_rate=3e-4
        )

        envmodel = EnvModel(
            p_observation_space=self._env._state_space,
            p_action_space=self._env._action_space,
            p_latency=self._env.get_latency(),
            p_afct_strans=afct_strans,
            p_afct_reward=self._env,
            p_afct_success=self._env,
            p_afct_broken=self._env,
            p_ada=p_ada,
            p_init_states=self._env.get_state(),
            p_logging=p_logging
        )

        mb_training_param = dict(p_cycle_limit=100,
                                 p_cycles_per_epi_limit=100,
                                 p_max_stagnations=0,
                                 p_collect_states=False,
                                 p_collect_actions=False,
                                 p_collect_rewards=False,
                                 p_collect_training=False)

        return Agent(
            p_policy=policy_random,  
            p_envmodel=envmodel,
            p_em_acc_thsld=0.8,
            p_action_planner=MPC(p_range_max=mt.Async.C_RANGE_THREAD,
                                 p_logging=p_logging),
            p_predicting_horizon=5,
            p_controlling_horizon=1,
            p_planning_width=50,
            p_name='Smith',
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging,
            **mb_training_param
        )
              

            
                
                
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# 2 Train agent in scenario
if __name__ == "__main__":
    # 2.1 Parameters for demo mode
    cycle_limit = 5000
    logging     = Log.C_LOG_ALL
    visualize   = True
    path        = str(Path.home())
    plotting    = True
else:
    # 2.2 Parameters for internal unit test
    cycle_limit = 10
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    path        = None
    plotting    = False

training = RLTraining(
    p_scenario_cls=ScenarioGridWorld,
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
              

            
                
                
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# 3 Plotting with MLpro
class MyDataPlotting(DataPlotting):


## -------------------------------------------------------------------------------------------------
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
                    raw.extend(self.data.get_values(name, fr_id))
                    if self.printing[name][1] == -1:
                        maxval = max(raw)
                        minval = min(raw)
                    else:
                        maxval = self.printing[name][2]
                        minval = self.printing[name][1]

                    label.append("%s" % fr_id)
                ax.plot(raw)
                ax.set_ylim(minval - (abs(minval) * 0.1), maxval + (abs(maxval) * 0.1))
                plt.xlabel("continuous cycles")
                self.plots[0].append(name)
                self.plots[1].append(ax)
                if self.showing:
                    plt.show()
                else:
                    plt.close(fig)
                          

                        
                            
                            
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    data_printing = {
        "Cycle": [False],
        "Day": [False],
        "Second": [False],
        "Microsecond": [False],
        "Smith": [True, -1],
    }
    
    mem = training.get_results().ds_rewards
    mem_plot = MyDataPlotting(mem, p_showing=plotting, p_printing=data_printing)
    mem_plot.get_plots()