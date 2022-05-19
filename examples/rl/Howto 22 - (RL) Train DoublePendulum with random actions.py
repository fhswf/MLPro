## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : Howto 22 - (RL) Train DoublePendulum with random actions.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-04-23  0.0.0     YI       Creation
## -- 2022-04-28  0.0.0     YI       Changing the Scenario and Debugging
## -- 2022-05-16  1.0.0     SY       Code cleaning, remove unnecessary, release the first version
## -------------------------------------------------------------------------------------------------


"""
Ver. 0.0.0 (2022-04-23)

This module shows how to use train the dpuble pendulum using random actions agent
"""

import torch
from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.pool.envs.doublependulum import DoublePendulum
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random
import os



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# 1. Create a policy for action sampling
class RandomActionGenerator(Policy):

    C_NAME      = 'RandomActionGenerator'
    
    
## -------------------------------------------------------------------------------------------------
    def __init__(self, p_observation_space:MSpace, p_action_space:MSpace, p_buffer_size=1, p_ada=1.0, p_logging=True):
        super().__init__(p_observation_space=p_observation_space, p_action_space=p_action_space, p_buffer_size=p_buffer_size, p_ada=p_ada, p_logging=p_logging)
        self.additional_buffer_element = {}
        self.action_current = np.zeros(p_action_space.get_num_dim())


## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self):
        pass

## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state: State) -> Action:
        self.action_current[0] = random.uniform(0,1)
        
        return Action(self._id, self._action_space, self.action_current)

## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        self.log(self.C_LOG_TYPE_I, 'No adaptation required')
        return False



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# 2 Implement the random RL scenario
class ScenarioDoublePendulum(RLScenario):

    C_NAME      = 'Matrix'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada, p_logging):
        # 1 Setup environment
        self._env   = DoublePendulum(p_logging=True, init_angles='up', max_torque=50)
        policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                     net_arch=[dict(pi=[128, 128], vf=[128, 128])])

        policy_random = RandomActionGenerator(p_observation_space=self._env.get_state_space(), 
                                              p_action_space=self._env.get_action_space(),
                                              p_buffer_size=1,
                                              p_ada=1,
                                              p_logging=False)



        # 2 Setup standard single-agent with own policy
        return Agent(
            p_policy=policy_random,  
            p_envmodel=None,
            p_name='smith',
            p_ada=p_ada,
            p_logging=p_logging
        )



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# 3 Create scenario and start training

if __name__ == "__main__":
    # 3.1 Parameters for demo mode
    cycle_limit         = 100
    adaptation_limit    = 10000
    stagnation_limit    = 0
    eval_frequency      = 5
    eval_grp_size       = 5
    logging             = Log.C_LOG_WE
    visualize           = True
    path                = str(Path.home())
    plotting        = True
 

# 4 Train agent in scenario 
now             = datetime.now()

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



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# 5 Create Plotting Class
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


#  Plotting 1 MLpro    
data_printing   = {"Cycle":        [False],
                    "Day":          [False],
                    "Second":       [False],
                    "Microsecond":  [False],
                    "Smith":        [True,-1]}


mem = training.get_results().ds_rewards
mem_plot    = MyDataPlotting(mem, p_showing=plotting, p_printing=data_printing)
mem_plot.get_plots()