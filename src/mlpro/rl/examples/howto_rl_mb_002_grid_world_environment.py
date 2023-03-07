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
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.0.0 (2023-03-07)

This module shows how to incorporate MPC in Model-Based RL on Grid World problem.

You will learn:
    
1) How to set up an own agent on Grid World problem
    
2) How to set up model-based RL (MBRL) training
    
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
from mlpro.rl.pool.envmodels.mlp_gridworld import MLPEnvModel
import mlpro.bf.mt as mt
              

            
                
                
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# 1 Set up MLP for Grid World
class GridWorldAFct(PyTorchMLP):
    
    C_NAME = "Grid World Adaptive Function"


## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self, **p_par): 
        self._hyperparam_space.add_dim(HyperParam('input_size','Z'))
        self._hyperparam_space.add_dim(HyperParam('output_size','Z'))
        self._hyperparam_space.add_dim(HyperParam('update_rate','Z'))
        self._hyperparam_space.add_dim(HyperParam('num_hidden_layers','Z'))
        self._hyperparam_space.add_dim(HyperParam('hidden_size','Z'))
        self._hyperparam_space.add_dim(HyperParam('activation_fct'))
        self._hyperparam_space.add_dim(HyperParam('output_activation_fct'))
        self._hyperparam_space.add_dim(HyperParam('optimizer'))
        self._hyperparam_space.add_dim(HyperParam('loss_fct'))
        self._hyperparam_space.add_dim(HyperParam('learning_rate','R'))
        self._hyperparam_tuple = HyperParamTuple(self._hyperparam_space)
        
        ids_ = self._hyperparam_tuple.get_dim_ids()
        self._hyperparam_tuple.set_value(ids_[0], self._input_space.get_num_dim())
        self._hyperparam_tuple.set_value(ids_[1], self._output_space.get_num_dim())
        self._hyperparam_tuple.set_value(ids_[2], 1)
        self._hyperparam_tuple.set_value(ids_[3], 3)
        self._hyperparam_tuple.set_value(ids_[4], 128)
        self._hyperparam_tuple.set_value(ids_[5], torch.nn.ReLU)
        self._hyperparam_tuple.set_value(ids_[6], torch.nn.ReLU)
        self._hyperparam_tuple.set_value(ids_[7], torch.optim.Adam)
        self._hyperparam_tuple.set_value(ids_[8], torch.nn.MSELoss)
        self._hyperparam_tuple.set_value(ids_[9], 3e-4)
              

            
                
                
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MLPEnvModel(EnvModel):
    C_NAME = "MLP Env Model for Grid World"
    
    
## -------------------------------------------------------------------------------------------------
    def __init__(
        self,
        p_ada=True,
        p_logging=False,
    ):

        self.grid_world = GridWorld(p_logging=p_logging,
                                    p_action_type=GridWorld.C_ACTION_TYPE_DISC_2D)
        
        # Setup Adaptive Function
        afct_strans = AFctSTrans(
            GridWorldAFct,
            p_state_space=self.grid_world._state_space,
            p_action_space=self.grid_world._action_space,
            p_threshold=1.8,
            p_buffer_size=5000,
            p_ada=p_ada,
            p_logging=p_logging,
        )

        EnvModel.__init__(
            self,
            p_observation_space=self.grid_world._state_space,
            p_action_space=self.grid_world._action_space,
            p_latency=timedelta(seconds=0.1),
            p_afct_strans=afct_strans,
            p_afct_reward=None,
            p_afct_success=None,
            p_afct_broken=None,
            p_ada=p_ada,
            p_logging=p_logging,
        )

        self.reset()
        
        
## -------------------------------------------------------------------------------------------------
    def get_state(self) -> State:
        return self.grid_world._state


## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old: State, p_state_new: State) -> Reward:
        return self.grid_world._compute_reward(p_state_old, p_state_new)


## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state:State) -> bool:
        return self.grid_world._compute_success(p_state)


## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state:State) -> bool:
        return self.grid_world._compute_broken(p_state)


## -------------------------------------------------------------------------------------------------
    def _process_action(self, p_action: Action) -> bool:
    
        self._set_state(self.simulate_reaction(self.get_state(), p_action))
    
        state = self.get_state()
        state.set_success(self.compute_success(state))
        state.set_broken(self.compute_broken(state))
        if state.get_broken() or state.get_success():
            state.set_terminal(True)
    
        return True
              

            
                
                
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# 2 Implement the random RL scenario
class ScenarioGridWorld(RLScenario):

    C_NAME      = 'Grid World with Random Actions'


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        # 2.1 Setup environment
        self._env   = GridWorld(p_logging=p_logging,
                                p_action_type=GridWorld.C_ACTION_TYPE_DISC_2D,
                                p_max_step=100)


        # 2.2 Setup and return random action agent
        policy_random = RandomGenerator(p_observation_space=self._env.get_state_space(), 
                                        p_action_space=self._env.get_action_space(),
                                        p_buffer_size=1,
                                        p_ada=1,
                                        p_logging=p_logging)

        mb_training_param = dict(p_cycle_limit=100,
                                 p_cycles_per_epi_limit=100,
                                 p_max_stagnations=0,
                                 p_collect_states=False,
                                 p_collect_actions=False,
                                 p_collect_rewards=False,
                                 p_collect_training=False)

        return Agent(
            p_policy=policy_random,  
            p_envmodel=MLPEnvModel(),
            p_em_acc_thsld=0.2,
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

# 3 Train agent in scenario
if __name__ == "__main__":
    # 3.1 Parameters for demo mode
    cycle_limit = 5000
    logging     = Log.C_LOG_ALL
    visualize   = True
    path        = str(Path.home())
    plotting    = True
else:
    # 3.2 Parameters for internal unit test
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