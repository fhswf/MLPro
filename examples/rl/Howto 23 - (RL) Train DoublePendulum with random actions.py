## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : Train DoublePendulum with Random Actions Agent
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-04-23  0.0.0     YI       Creation
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
from mlpro.wrappers.sb3 import WrPolicySB32MLPro



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
        
        _id = self._action_space.get_dim_ids()[0]
        if self._action_space.get_dim(_id).get_base_set() == "Z":
            if self.action_current[0] >= 0.5:
                self.action_current[0] = 1
            else:
                self.action_current[0] = 0
                
        return Action(self._id, self._action_space, self.action_current)

## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        self.log(self.C_LOG_TYPE_I, 'No adaptation required')
        return False

# 2 Implement the random RL scenario
class ScenarioDoublePendulum(RLScenario):

    C_NAME      = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):
        self._env       = DoublePendulum(p_logging=True)
        state_space     = self._env.get_state_space()
        action_space    = self._env.get_action_space()
        
        mb_training_param = dict(p_cycle_limit=100,
                                 p_cycles_per_epi_limit=100,
                                 p_max_stagnations=0,
                                 p_collect_states=False,
                                 p_collect_actions=False,
                                 p_collect_rewards=False,
                                 p_collect_training=False)
        
        
        # Agent 1
        _name         = 'random_actions_agent'
        _ospace       = state_space.spawn([state_space.get_dim_ids()[0],state_space.get_dim_ids()[1]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[0]])
        _policy       = RandomActionGenerator(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
        self._agent.add_agent(
            p_agent=MyAgent(
                p_policy=_policy,
                p_envmodel=MLPEnvModel(_ospace,_aspace),
                p_em_mat_thsld=-1,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True,
                **mb_training_param),
            p_weight=1.0
            )
        return self._agent
        

# 2 Create scenario and start training

if __name__ == "__main__":
    # 3.1 Parameters for demo mode
    cycle_limit         = 0
    adaptation_limit    = 6000
    stagnation_limit    = 0
    eval_frequency      = 5
    eval_grp_size       = 5
    logging             = Log.C_LOG_WE
    visualize           = True
    path                = str(Path.home())
    plotting        = True
 

# 3 Train agent in scenario 
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