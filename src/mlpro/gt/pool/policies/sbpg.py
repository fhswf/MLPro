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

Here are the related papers for citations:
    - Best response learning (DOI: 10.1109/TCYB.2020.3006620)
    - Gradient-based learning (DOI: 10.1109/IECON55916.2024.10905619)
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
        
        self.C_SCIREF_TYPE      = self.C_SCIREF_TYPE_INPROCEEDINGS
        self.C_SCIREF_AUTHOR    = "Steve Yuwono, Marlon Löppenberg, Dorothea Schwung, Andreas Schwung"
        self.C_SCIREF_BOOKTITLE = "IECON 2024 - 50th Annual Conference of the IEEE Industrial Electronics Society"
        self.C_SCIREF_TITLE     = "Gradient-based Learning in State-based Potential Games for Self-Learning Production Systems"
        self.C_SCIREF_YEAR      = "2024"
        self.C_SCIREF_PAGES     = "1-7"
        self.C_SCIREF_DOI       = "10.1109/IECON55916.2024.10905619"

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
        
        self.performance_map     = torch.zeros((2, num_states, num_states))
        x                        = torch.linspace(1/(num_states*2), 1-(1/(num_states*2)), num_states)
        self.grid_x, self.grid_y = torch.meshgrid(x, x)
        self.exploration         = 1
        self._counter            = 0
        self._current_ep         = 1
        
        if self._algo != self.ALG_SbPG_BR:
            self.map_nxt_action  = torch.zeros((num_states, num_states))
        
        
## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self):
        
        self._hyperparam_space.add_dim(HyperParam('num_states','Z'))
        self._hyperparam_space.add_dim(HyperParam('exploration_decay','R'))
        self._hyperparam_space.add_dim(HyperParam('alpha','R'))             # GB learning
        self._hyperparam_space.add_dim(HyperParam('OU_noise','R'))          # GB learning
        self._hyperparam_space.add_dim(HyperParam('kick_off_eps','Z'))      # GB learning
        self._hyperparam_space.add_dim(HyperParam('cycles_per_ep','Z'))
        self._hyperparam_space.add_dim(HyperParam('smoothing','R'))
        self._hyperparam_space.add_dim(HyperParam('ep_max','Z'))
        self._hyperparam_space.add_dim(HyperParam('beta','R'))              # GB learning (Momentum)
        self._hyperparam_tuple = HyperParamTuple(self._hyperparam_space)
        
        ids_ = self._hyperparam_tuple.get_dim_ids()
        self._hyperparam_tuple.set_value(ids_[0], 40)
        self._hyperparam_tuple.set_value(ids_[1], 0.55**(2/1000))
        self._hyperparam_tuple.set_value(ids_[2], 1.0)
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
    def compute_action(self, p_state:State) -> Action:
        
        if self._algo == self.ALG_SbPG_BR:
            return self.compute_action_br(p_state)
        else:
            return self.compute_action_gb(p_state)


## -------------------------------------------------------------------------------------------------
    def compute_action_br(self, p_state:State) -> Action:
        
        my_action_values        = np.zeros(self._action_space.get_num_dim())
        actual_states           = p_state.get_values()
        
        if self._counter == self.get_hyperparam().get_value(self._hp_ids[5]):
            self._counter       = 0
            self.exploration    = 1
            self._current_ep    += 1
        
        if self._current_ep == self.get_hyperparam().get_value(self._hp_ids[7]):
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
    def compute_action_gb(self, p_state:State) -> Action:
        
        my_action_values        = np.zeros(self._action_space.get_num_dim())
        actual_states           = p_state.get_values()
        x_disc, y_disc          = self._discretization(actual_states[0],actual_states[1])
    
        if self._counter == self.get_hyperparam().get_value(self._hp_ids[5]):
            self._counter       = 0
            self.exploration    = 1
            self._current_ep    += 1
        
        if self._current_ep == self.get_hyperparam().get_value(self._hp_ids[7]):
            self.exploration    = 0
            
        if random.uniform(0, 1) < self.exploration:
            if self._current_ep <= self.get_hyperparam().get_value(self._hp_ids[4]):
                my_action_values[0] = random.uniform(0, 1)
            else:
                nxt_action          = self.map_nxt_action[x_disc, y_disc].item()
                ou_noise            = self.get_hyperparam().get_value(self._hp_ids[3])
                my_action_values[0] = max(0,min(1,random.uniform(nxt_action-ou_noise,nxt_action+ou_noise)))
        else:
            my_action_values[0]     = self.interpolate_maps(actual_states[0], actual_states[1])
            
        _id = self._action_space.get_dim_ids()[0]
        if self._action_space.get_dim(_id).get_base_set() == "Z":
            if my_action_values[0] >= 0.5:
                my_action_values[0] = 1
            else:
                my_action_values[0] = 0
                
        return Action(self._id, self._action_space, my_action_values)


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        
        if self._algo == self.ALG_SbPG_BR:
            return self._adapt_br(p_kwargs)
        elif self._algo == self.ALG_SbPG_GB:
            return self._adapt_gb(p_kwargs)
        elif self._algo == self.ALG_SbPG_GB_MOM:
            return self._adapt_gb_mom(p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _adapt_br(self, p_kwargs) -> bool:
        
        prev_states         = p_kwargs["p_sars_elem"].get_data()["state"].get_values()
        x_disc, y_disc      = self._discretization(prev_states[0], prev_states[1])
        
        self.exploration    *= self.get_hyperparam().get_value(self._hp_ids[1])
        self._counter       += 1

        rwd                 = p_kwargs["p_sars_elem"].get_data()["reward"].get_agent_reward(self._id)[0]
        if rwd > (self.performance_map[1, x_disc, y_disc]).item():
            self.performance_map[1,x_disc,y_disc] = rwd
            self.performance_map[0,x_disc,y_disc] = p_kwargs["p_sars_elem"].get_data()["action"].get_sorted_values().item()
            return True
        else:
            return False


## -------------------------------------------------------------------------------------------------
    def _adapt_gb(self, p_kwargs) -> bool:
        
        prev_states     = p_kwargs["p_sars_elem"].get_data()["state"].get_values()
        x_disc, y_disc  = self._discretization(prev_states[0], prev_states[1])
        prev_action     = self.performance_map[0, x_disc, y_disc]
        prev_util       = self.performance_map[1, x_disc, y_disc]
        
        act             = p_kwargs["p_sars_elem"].get_data()["action"].get_sorted_values().item()
        util            = p_kwargs["p_sars_elem"].get_data()["reward"].get_agent_reward(self._id)[0]
        
        if (act-prev_action) == 0:
            gradient    = (util-prev_util)
        else:
            gradient    = (util-prev_util)/(act-prev_action)
        if util > self.performance_map[1, x_disc, y_disc]:
            self.performance_map[0, x_disc, y_disc] = act
            self.performance_map[1, x_disc, y_disc] = util
            
        lr_gradient     = self.get_hyperparam().get_value(self._hp_ids[2])
        self.map_nxt_action[x_disc, y_disc] = min(1,max(0,act)+(lr_gradient*gradient))
        
        self.exploration *= self.get_hyperparam().get_value(self._hp_ids[1])
        self._counter += 1
        
        return True

## -------------------------------------------------------------------------------------------------
    def _adapt_gb_mom(self, p_kwargs) -> bool:
        
        prev_states     = p_kwargs["p_sars_elem"].get_data()["state"].get_values()
        x_disc, y_disc  = self._discretization(prev_states[0], prev_states[1])
        prev_action     = self.performance_map[0, x_disc, y_disc]
        prev_util       = self.performance_map[1, x_disc, y_disc]
        
        act             = p_kwargs["p_sars_elem"].get_data()["action"].get_sorted_values().item()
        util            = p_kwargs["p_sars_elem"].get_data()["reward"].get_agent_reward(self._id)[0]
               
        if (act-prev_action) == 0:
            gradient    = (util-prev_util)
        else:
            gradient    = (util-prev_util)/(act-prev_action)
        if util > self.performance_map[1, x_disc, y_disc]:
            self.performance_map[0, x_disc, y_disc] = act
            self.performance_map[1, x_disc, y_disc] = util
            
        lr_gradient     = self.get_hyperparam().get_value(self._hp_ids[2])
        if self.map_nxt_action[x_disc, y_disc] == 0:
            self.map_nxt_action[x_disc, y_disc] = min(1,max(0,act)+(lr_gradient*gradient))
        else:
            prev_gradient = self.map_nxt_action[x_disc, y_disc]
            gradient = self.get_hyperparam().get_value(self._hp_ids[8])*prev_gradient + (1-self.get_hyperparam().get_value(self._hp_ids[8]))*gradient
            self.map_nxt_action[x_disc, y_disc] = min(1,max(0,act)+(lr_gradient*gradient))
            
        self.exploration *= self.get_hyperparam().get_value(self._hp_ids[1])
        self._counter += 1
        
        return True
        
        
## -------------------------------------------------------------------------------------------------        
    def _discretization(self, p_x_fill_level, p_y_fill_level):
               
        num_states  = self.get_hyperparam().get_value(self._hp_ids[0])
        x_disc      = min(m.floor(p_x_fill_level*num_states), int(num_states-1))
        y_disc      = min(m.floor(p_y_fill_level*num_states), int(num_states-1))
        
        return x_disc, y_disc
    
    
## -------------------------------------------------------------------------------------------------    
    def interpolate_maps(self, p_pos_x, p_pos_y):
                
        distances = torch.sqrt(((p_pos_x-self.grid_x)**2) + ((p_pos_y-self.grid_y)**2))
        distances[distances == 0] = 0.0001
        weights = 1/((distances**2)+self.get_hyperparam().get_value(self._hp_ids[6]))
        DistancesTotal = sum(sum(weights))
        outputs = (weights/DistancesTotal)*self.performance_map[0,:,:]

        return sum(sum(outputs))