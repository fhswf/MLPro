## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.gt.pool.policies
## -- Module  : sbpg.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-04-09  0.0.0     SY       Creation
## -- 2025-04-09  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-04-09)

This module implements a state-based potential games (SbPG) with best response and two types of 
gradient-based learning for dynamic games.
"""

from mlpro.rl.models import *
from mlpro.bf.ml import *
import random
import math as m
import torch
         
        



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SbPG(Policy):
    
    C_NAME          = 'SbPG'
    
    ALG_SbPG_BR     = 0
    ALG_SbPG_GB     = 1
    ALG_SbPG_GB_MOM = 2


## -------------------------------------------------------------------------------------------------
    def __init__(
            self,
            p_observation_space:MSpace,
            p_action_space:MSpace,
            p_id:str=None,
            p_buffer_size:int=1,
            p_ada:bool=True,
            p_visualize:bool=False,
            p_logging=Log.C_LOG_ALL,
            p_algo:int=0,
            p_num_states:int=None,
            p_exploration_decay:float=None,
            p_alpha:float=None,
            p_ou_noise:float=None,
            p_kick_off_eps:int=None,
            p_cycles_per_ep:int=None,
            p_smoothing:float=None,
            p_ep_max:int=None,
            p_beta:float=None
            ):
        
        Policy.__init__(
            self,
            p_observation_space=p_observation_space,
            p_action_space=p_action_space,
            p_id=p_id,
            p_buffer_size=p_buffer_size,
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging
            )
        
        self._hyperparam_space  = HyperParamSpace()
        self._hyperparam_tuple  = None
        self._init_hyperparam()
    	
        ids_                    = self._hyperparam_tuple.get_dim_ids()
        if p_num_states is not None:
            self._hyperparam_tuple.set_value(ids_[0], p_num_states)
        if p_exploration_decay is not None:
            self._hyperparam_tuple.set_value(ids_[1], p_exploration_decay)
        if p_alpha is not None:
            self._hyperparam_tuple.set_value(ids_[2], p_alpha)
        if p_ou_noise is not None:
            self._hyperparam_tuple.set_value(ids_[3], p_ou_noise)
        if p_kick_off_eps is not None:
            self._hyperparam_tuple.set_value(ids_[4], p_kick_off_eps)
        if p_cycles_per_ep is not None:
            self._hyperparam_tuple.set_value(ids_[5], p_cycles_per_ep)
        if p_smoothing is not None:
            self._hyperparam_tuple.set_value(ids_[6], p_smoothing)
        if p_ep_max is not None:
            self._hyperparam_tuple.set_value(ids_[7], p_ep_max)
        if p_beta is not None:
            self._hyperparam_tuple.set_value(ids_[8], p_beta)

        self._hp_ids            = self.get_hyperparam().get_dim_ids()
        num_states              = int(self.get_hyperparam().get_value(self._hp_ids[0]))
        self._algo              = p_algo
        
        self.exploration         = 1
        self.performance_map     = torch.zeros((2, num_states, num_states))
        x                        = torch.linspace(1/(num_states*2), 1-(1/(num_states*2)), num_states)
        self.grid_x, self.grid_y = torch.meshgrid(x, x)
        self._counter            = 0
        self._current_ep         = 1
        
        if self._algo != self.ALG_Vanilla_SbPG:
            self.map_nxt_action  = torch.zeros((num_states, num_states))
        
        
## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self):
        # 1.2 Declare hyperparameters with unique id, names, and data type   
        self._hyperparam_space.add_dim(HyperParam('num_states','Z'))
        self._hyperparam_space.add_dim(HyperParam('exploration_decay','R'))
        self._hyperparam_space.add_dim(HyperParam('alpha','R'))                 # gradient-based learning
        self._hyperparam_space.add_dim(HyperParam('OU_noise','R'))              # gradient-based learning
        self._hyperparam_space.add_dim(HyperParam('kick_off_eps','Z'))          # gradient-based learning
        self._hyperparam_space.add_dim(HyperParam('cycles_per_ep','Z'))
        self._hyperparam_space.add_dim(HyperParam('smoothing','R'))
        self._hyperparam_space.add_dim(HyperParam('ep_max','Z'))
        self._hyperparam_space.add_dim(HyperParam('beta','R'))                 # gradient-based learning (Method 2)
        self._hyperparam_tuple = HyperParamTuple(self._hyperparam_space)
        
        # 1.3 Set the hyperparameter with a default value 
        ids_ = self._hyperparam_tuple.get_dim_ids()
        self._hyperparam_tuple.set_value(ids_[0], 40)
        self._hyperparam_tuple.set_value(ids_[1], 0.55**(2/1000))
        self._hyperparam_tuple.set_value(ids_[2], 1.0) # gb method 1 = 1.0, method 2 = 0.5
        self._hyperparam_tuple.set_value(ids_[3], 0)
        self._hyperparam_tuple.set_value(ids_[4], 0)
        self._hyperparam_tuple.set_value(ids_[5], 1000)
        self._hyperparam_tuple.set_value(ids_[6], 0.000035)
        self._hyperparam_tuple.set_value(ids_[7], 10)
        self._hyperparam_tuple.set_value(ids_[8], 0.4)
        
        
## -------------------------------------------------------------------------------------------------
    def get_hyperparam(self) -> HyperParamTuple:
        return self._hyperparam_tuple


## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state: State) -> Action:
        
        if self._algo == self.ALG_Vanilla_SbPG:
            return self.compute_action_1(p_state)
        else:
            return self.compute_action_2(p_state)


## -------------------------------------------------------------------------------------------------
    def compute_action_1(self, p_state: State) -> Action:
        
        my_action_values    = np.zeros(self._action_space.get_num_dim())
        actual_states       = p_state.get_values()
        ids_                = self.get_hyperparam().get_dim_ids()
        
        x_disc, y_disc      = self._discretization(
            actual_states[0],
            actual_states[1]
            )
        
        if self._counter == self.get_hyperparam().get_value(ids_[5]):
            self._counter       = 0
            self.exploration    = 1
            self._current_ep    += 1
        
        if self._current_ep == self.get_hyperparam().get_value(ids_[7]):
            self.exploration    = 0
            
        if random.uniform(0, 1) < self.exploration:
            my_action_values[0] = random.uniform(0, 1)
        else:
            my_action_values[0] = self.interpolate_maps(actual_states[0], actual_states[1])

        _id = self._action_space.get_dim_ids()[0]
        
        if self._action_space.get_dim(_id).get_base_set() == "Z":
            if my_action_values[0] >= 0.5:
                my_action_values[0] = 1
            else:
                my_action_values[0] = 0
                
        return Action(self._id, self._action_space, my_action_values)


## -------------------------------------------------------------------------------------------------
    def compute_action_2(self, p_state: State) -> Action:
        
        my_action_values    = np.zeros(self._action_space.get_num_dim())
        actual_states       = p_state.get_values()
        ids_                = self.get_hyperparam().get_dim_ids()
        
        x_disc, y_disc      = self._discretization(
            actual_states[0],
            actual_states[1]
            )
    
        if self._counter == self.get_hyperparam().get_value(ids_[5]):
            self._counter       = 0
            self.exploration    = 1
            self._current_ep    += 1
        
        if self._current_ep == self.get_hyperparam().get_value(ids_[7]):
            self.exploration    = 0
            
        if random.uniform(0, 1) < self.exploration:
            if self._current_ep <= self.get_hyperparam().get_value(ids_[4]):
                my_action_values[0] = random.uniform(0, 1)
            else:
                nxt_action = self.map_nxt_action[x_disc, y_disc].item()
                ou_noise = self.get_hyperparam().get_value(ids_[3])
                my_action_values[0] = max(0,min(1,random.uniform(nxt_action-ou_noise,nxt_action+ou_noise)))
        else:
            # if self.exploration != 0 and self.performance_map[0, x_disc, y_disc] and actual_states[0] <= 0.8:
            #     my_action_values[0] = random.uniform(0, 1)
            # else:
            #     my_action_values[0] = self.interpolate_maps(actual_states[0], actual_states[1])
            my_action_values[0] = self.interpolate_maps(actual_states[0], actual_states[1])
            
        _id = self._action_space.get_dim_ids()[0]
        
        if self._action_space.get_dim(_id).get_base_set() == "Z":
            if my_action_values[0] >= 0.5:
                my_action_values[0] = 1
            else:
                my_action_values[0] = 0
                
        return Action(self._id, self._action_space, my_action_values)


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        
        if self._algo == self.ALG_Vanilla_SbPG:
            return self._adapt_1(p_kwargs)
        elif self._algo == self.ALG_GB_SbPG_METHOD_1:
            return self._adapt_2(p_kwargs)
        elif self._algo == self.ALG_GB_SbPG_METHOD_2:
            return self._adapt_3(p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _adapt_1(self, p_kwargs) -> bool:
        
        prev_states = p_kwargs["p_sars_elem"].get_data()["state"].get_values()
        
        x_disc, y_disc = self._discretization(
            prev_states[0],
            prev_states[1]
            )
        
        rwd = p_kwargs["p_sars_elem"].get_data()["reward"].get_agent_reward(self._id)[0]
        
        if rwd > (self.performance_map[1,x_disc,y_disc]).item():
            self.performance_map[1,x_disc,y_disc] = rwd
            self.performance_map[0,x_disc,y_disc] = p_kwargs["p_sars_elem"].get_data()["action"].get_sorted_values().item()
        
        ids_ = self.get_hyperparam().get_dim_ids()
        self.exploration *= self.get_hyperparam().get_value(ids_[1])
        self._counter += 1
        
        return True


## -------------------------------------------------------------------------------------------------
    def _adapt_2(self, p_kwargs) -> bool:
        
        prev_states = p_kwargs["p_sars_elem"].get_data()["state"].get_values()
        
        x_disc, y_disc = self._discretization(
            prev_states[0],
            prev_states[1]
            )
        
        prev_action = self.performance_map[0, x_disc, y_disc]
        prev_util = self.performance_map[1, x_disc, y_disc]
        
        act = p_kwargs["p_sars_elem"].get_data()["action"].get_sorted_values().item()
        util = p_kwargs["p_sars_elem"].get_data()["reward"].get_agent_reward(self._id)[0]
               
        if (act-prev_action) == 0:
            gradient = (util-prev_util)
        else:
            gradient = (util-prev_util)/(act-prev_action)
        if util > self.performance_map[1, x_disc, y_disc]:
            self.performance_map[0, x_disc, y_disc] = act
            self.performance_map[1, x_disc, y_disc] = util
            
        ids_ = self.get_hyperparam().get_dim_ids()
        lr_gradient = self.get_hyperparam().get_value(ids_[2])
        self.map_nxt_action[x_disc, y_disc] = min(1,max(0,act)+(lr_gradient*gradient))
        
        self.exploration *= self.get_hyperparam().get_value(ids_[1])
        self._counter += 1
        
        return True

## -------------------------------------------------------------------------------------------------
    def _adapt_3(self, p_kwargs) -> bool:
        
        prev_states = p_kwargs["p_sars_elem"].get_data()["state"].get_values()
        
        x_disc, y_disc = self._discretization(
            prev_states[0],
            prev_states[1]
            )
        
        prev_action = self.performance_map[0, x_disc, y_disc]
        prev_util = self.performance_map[1, x_disc, y_disc]
        
        act = p_kwargs["p_sars_elem"].get_data()["action"].get_sorted_values().item()
        util = p_kwargs["p_sars_elem"].get_data()["reward"].get_agent_reward(self._id)[0]
               
        if (act-prev_action) == 0:
            gradient = (util-prev_util)
        else:
            gradient = (util-prev_util)/(act-prev_action)
        if util > self.performance_map[1, x_disc, y_disc]:
            self.performance_map[0, x_disc, y_disc] = act
            self.performance_map[1, x_disc, y_disc] = util
            
        ids_ = self.get_hyperparam().get_dim_ids()
        lr_gradient = self.get_hyperparam().get_value(ids_[2])
        if self.map_nxt_action[x_disc, y_disc] == 0:
            self.map_nxt_action[x_disc, y_disc] = min(1,max(0,act)+(lr_gradient*gradient))
        else:
            prev_gradient = self.map_nxt_action[x_disc, y_disc]
            gradient = self.get_hyperparam().get_value(ids_[8])*prev_gradient + (1-self.get_hyperparam().get_value(ids_[8]))*gradient
            self.map_nxt_action[x_disc, y_disc] = min(1,max(0,act)+(lr_gradient*gradient))
            
        self.exploration *= self.get_hyperparam().get_value(ids_[1])
        self._counter += 1
        
        return True
        
        
## -------------------------------------------------------------------------------------------------        
    def _discretization(self, x_fill_level, y_fill_level):
               
        ids_        = self.get_hyperparam().get_dim_ids()
        num_states  = self.get_hyperparam().get_value(ids_[0])
        x_disc      = min(m.floor(x_fill_level*num_states), int(num_states-1))
        y_disc      = min(m.floor(y_fill_level*num_states), int(num_states-1))
        
        return x_disc, y_disc
    
    
## -------------------------------------------------------------------------------------------------    
    def interpolate_maps(self, pos_x, pos_y):
        
        ids_ = self.get_hyperparam().get_dim_ids()
        
        distances = torch.sqrt(((pos_x-self.grid_x)**2) + ((pos_y-self.grid_y)**2))
        distances[distances == 0] = 0.0001
        weights = 1/((distances**2)+self.get_hyperparam().get_value(ids_[6]))
        DistancesTotal = sum(sum(weights))
        outputs = (weights/DistancesTotal)*self.performance_map[0,:,:]
        return sum(sum(outputs))