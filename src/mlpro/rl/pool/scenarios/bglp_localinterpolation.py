## -----------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : bglp_localinterpolation
## -----------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.  Auth.  Description
## -- 2021-09-06  0.00  SY     Creation
## -- 2021-09-06  1.00  SY     Release of first version
## -- 2021-09-11  1.00  MRD    Change Header information to match our new library name
## -- 2021-09-22  1.01  SY     Solving minor bugs
## -- 2021 09-26  1.02  MRD    Change the import module due to the change of the pool
## --                          folder structer
## -- 2021 09-30  1.03  SY     Minor Improvements
## -----------------------------------------------------------------------------

"""
Ver. 1.03 (2021-09-30)

Environment : BGLP
Algorithms  : SbPG - Local Interpolation (dummy)
"""


from mlpro.rl.pool.envs import BGLP
from mlpro.rl.models import *
from mlpro.bf.various import *
from mlpro.bf.math import *
from mlpro.bf.ml import *
from mlpro.rl import *
import random
import numpy as np
import math as m
import torch

#################################################################

cycle_limit         = 1000

class MyPolicy(Policy):

    C_NAME      = 'MyPolicy'

    def __init__(self, p_state_space:MSpace, p_action_space:MSpace, p_buffer_size=1, p_ada=1.0, p_logging=True, episodes_max=10):
        super().__init__(p_state_space=p_state_space, p_action_space=p_action_space, p_ada=p_ada, p_logging=p_logging)
        self._state_space       = p_state_space
        self._action_space      = p_action_space
        self.set_id(0)
        
        self.levels_current     = np.zeros(p_state_space.get_num_dim())
        self.levels_current_con = np.zeros(p_state_space.get_num_dim())
        self.levels_last        = np.zeros(p_state_space.get_num_dim())
        self.levels_last_con    = np.zeros(p_state_space.get_num_dim())
        self.action_last        = np.zeros(p_action_space.get_num_dim())
        self.action_current     = np.zeros(p_action_space.get_num_dim())
        self.updated            = False
        self.ep                 = -1
        self.exploration        = 1
        self.ep_max             = episodes_max
        self.reset_exploration()

    def _init_hyperparam(self):
        self._hyperparam_space.add_dim(HyperParam(0,'num_states','Z'))
        self._hyperparam_space.add_dim(HyperParam(1,'smoothing','R'))
        self._hyperparam_space.add_dim(HyperParam(2,'maxrange','R'))
        self._hyperparam_space.add_dim(HyperParam(3,'rangecoeff','R'))
        self._hyperparam_space.add_dim(HyperParam(4,'lvl_max_silo','R'))
        self._hyperparam_space.add_dim(HyperParam(5,'lvl_max_hopper','R'))
        self._hyperparam_space.add_dim(HyperParam(6,'exp_decay','R'))
        self._hyperparam_tupel = HyperParamTupel(self._hyperparam_space)
        
        self._hyperparam_tupel.set_value(0, 40)
        self._hyperparam_tupel.set_value(1, 0.000035)
        self._hyperparam_tupel.set_value(2, 0.40)
        self._hyperparam_tupel.set_value(3, 0.5)
        self._hyperparam_tupel.set_value(4, 17.42)
        self._hyperparam_tupel.set_value(5, 9.10)
        self._hyperparam_tupel.set_value(6, 0.99998)

        self.num_states         = self._hyperparam_tupel.get_value(0)
        self.levels_max         = [self._hyperparam_tupel.get_value(4),
                                   self._hyperparam_tupel.get_value(5),
                                   self._hyperparam_tupel.get_value(4),
                                   self._hyperparam_tupel.get_value(5),
                                   self._hyperparam_tupel.get_value(4),
                                   self._hyperparam_tupel.get_value(5)]
        self.exp_decay          = self._hyperparam_tupel.get_value(6)
        self.smoothing          = self._hyperparam_tupel.get_value(1)
        self.maxrange           = self._hyperparam_tupel.get_value(2)
        self.rangecoeff         = self._hyperparam_tupel.get_value(3)
        self.map_utility    = torch.zeros(int(self.num_states), int(self.num_states))
        self.map_action     = torch.zeros(int(self.num_states), int(self.num_states))
        self.grid           = (torch.arange(int(self.num_states)).float()+1)/(int(self.num_states))-1/(int(self.num_states)*2)
        self.grid_center_x  = torch.zeros(int(self.num_states), int(self.num_states))
        self.grid_center_y  = torch.zeros(int(self.num_states), int(self.num_states))
        for x in range(int(self.num_states)):
            for y in range(int(self.num_states)):
                self.grid_center_x[x,y] = self.grid[x]
                self.grid_center_y[x,y] = self.grid[y]
    
    def calc_current_states(self, levels):
        for i in range(self._state_space.get_num_dim()):
            levels_cur = levels[i]*self.levels_max[i]
            self.levels_current[i] = min(m.floor(self.num_states*levels_cur/self.levels_max[i]),self.num_states-1)
    
    def memorize_levels(self):
        for i in range(self._state_space.get_num_dim()):
            self.levels_last[i] = self.levels_current[i]
            self.levels_last_con[i] = self.levels_current_con[i]
    
    def compute_action(self, p_state: State) -> Action:
        states = p_state.get_values()
        for i in range(self._state_space.get_num_dim()):
            self.levels_current_con[i] = states[i]
        self.calc_current_states(states)
        if random.uniform(0,1) <= self.exploration:
            self.action_current[0] = random.uniform(0,1)
        else:
            self.action_current[0] = self.interpolate_map(states[0],states[1])
        self.action_last[0] = self.action_current[0]
        self.memorize_levels()
        self.exploration = self.exploration*self.exp_decay
        return Action(self._id, self._action_space, self.action_current)
    
    def reset_exploration(self):
        self.ep += 1
        if self.ep >= self.ep_max:
            self.exploration = 0
        else:
            self.exploration = 1

    def adapt(self, *p_args) -> bool:
        if not super().adapt(*p_args):
            return False
        
        # if not self._buffer.is_full():
        #     return False
        
        # sar_data = self._buffer.get_all()
        # for reward in sar_data["reward"]:
        #     rwd = reward.get_agent_reward(self._id)
        # if self.updated:
        #     self.int_act()
        #     self.update_maps(self.action_last[0], rwd, self.levels_last, self.levels_last_con)
        # self.log(self.C_LOG_TYPE_I, 'Performance map is updated')
        return False
    
    def update_maps(self, action, utility, levels, levels_con):
        if utility > self.map_utility[levels[1], levels[0]]:
            self.update_area(levels, levels_con, action, utility)
    
    def update_area(self, levels, levels_con, action, utility):
        self.map_action[levels[1], levels[0]] = action
        self.map_utility[levels[1], levels[0]] = utility
        
          
    def interpolate_map(self, pos_y, pos_x):
        distances = torch.sqrt((pos_x.item()-self.grid_center_x)**2+(pos_y.item()-self.grid_center_y)**2)
        distances[distances == 0] = 0.0001
        ranges = (distances < self.maxrange ** 2).float()
        weight_vab = torch.nonzero(ranges).long()
        weights = torch.zeros(len(weight_vab),1)
        maps_update = torch.zeros(len(weight_vab),1)
        for x in range(len(weight_vab)):
            weights[x] = 1/(distances[weight_vab[x][0],weight_vab[x][1]]**2)+self.smoothing
            maps_update[x] = self.map_action[weight_vab[x][0],weight_vab[x][1]]
        DistancesTotal = sum(sum(weights))
        outputs = weights/DistancesTotal*maps_update
        return sum(sum(outputs))

    def clear_buffer(self):
        self._buffer.clear()
    
    def _add_additional_buffer(self, p_buffer_element: SARBufferElement):
        p_buffer_element.add_value_element(self.additional_buffer_element)
        return p_buffer_element
        
#################################################################

class MyScenario(Scenario):

    C_NAME      = 'BGLP_Environement'

    def _setup(self, p_mode, p_ada, p_logging):
        self._env       = BGLP(p_logging=True)
        self._agent     = MultiAgent(p_name='SbPG - Local Interpolation', p_ada=1, p_logging=False)
        state_space     = self._env.get_state_space()
        action_space    = self._env.get_action_space()
        
        
        # Agent 1
        agent_name      = 'BELT_CONVEYOR_A'
        agent_id        = 0
        agent_sspace    = state_space.spawn([0,1])
        agent_aspace    = action_space.spawn([0])
        agent_policy    = MyPolicy(p_state_space=agent_sspace, p_action_space=agent_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=agent_policy,
                p_envmodel=None,
                p_name=agent_name,
                p_id=agent_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Agent 2
        agent_name      = 'VACUUM_PUMP_B'
        agent_id        = 1
        agent_sspace    = state_space.spawn([1,2])
        agent_aspace    = action_space.spawn([1])
        agent_policy    = MyPolicy(p_state_space=agent_sspace, p_action_space=agent_aspace, p_ada=1, p_logging=False, p_buffer_size=1)
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=agent_policy,
                p_envmodel=None,
                p_name=agent_name,
                p_id=agent_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Agent 3
        agent_name      = 'VIBRATORY_CONVEYOR_B'
        agent_id        = 2
        agent_sspace    = state_space.spawn([2,3])
        agent_aspace    = action_space.spawn([2])
        agent_policy    = MyPolicy(p_state_space=agent_sspace, p_action_space=agent_aspace, p_ada=1, p_logging=False, p_buffer_size=1)
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=agent_policy,
                p_envmodel=None,
                p_name=agent_name,
                p_id=agent_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Agent 4
        agent_name      = 'VACUUM_PUMP_C'
        agent_id        = 3
        agent_sspace    = state_space.spawn([3,4])
        agent_aspace    = action_space.spawn([3])
        agent_policy    = MyPolicy(p_state_space=agent_sspace, p_action_space=agent_aspace, p_ada=1, p_logging=False, p_buffer_size=1)
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=agent_policy,
                p_envmodel=None,
                p_name=agent_name,
                p_id=agent_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Agent 5
        agent_name      = 'ROTARY_FEEDER_C'
        agent_id        = 4
        agent_sspace    = state_space.spawn([4,5])
        agent_aspace    = action_space.spawn([4])
        agent_policy    = MyPolicy(p_state_space=agent_sspace, p_action_space=agent_aspace, p_ada=1, p_logging=False, p_buffer_size=1)
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=agent_policy,
                p_envmodel=None,
                p_name=agent_name,
                p_id=agent_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
#################################################################


myscenario  = MyScenario(p_mode=Environment.C_MODE_SIM, p_ada=True, p_cycle_limit=cycle_limit,
                         p_visualize=True, p_logging=True)
