## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.rl.pool.actionplanner
## -- Module  : mpc
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-15  0.0.0     SY       Creation
## -- 2022-08-15  1.0.0     SY       Release of first version
## -- 2022-10-08  1.0.1     SY       Bug fixing
## -- 2023-01-02  1.1.0     SY       Add multiprocessing functionality
## -- 2023-02-04  1.1.1     SY       Bug fixing
## -- 2025-04-24  1.1.2     DA       Bugfix in method MPC._async_subtask()
## -- 2025-07-17  1.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2025-07-17) 

This module provides a default implementation of model predictive control (MPC).
"""

import random

import numpy as np

from mlpro.bf import Log, ParamError
from mlpro.bf.various import ScientificObject
from mlpro.bf.systems import State, Action 
import mlpro.bf.mt as mt

from mlpro.rl.models import *

        

# Export list for public API
__all__ = [ 'MPC' ]


        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MPC(ActionPlanner, ScientificObject, mt.Async):
    """
    Template class for MPC to be used as part of model-based planning agents. 
    The goal is to find the best sequence of actions that leads to a maximum reward.

    Parameters
    ----------
    p_range : int
        Range of asynchonicity.
    p_state_thsld : float
        Threshold for metric difference between two states to be equal. Default = 0.00000001.
    p_logging
        Log level (see constants of class Log). Default = Log.C_LOG_ALL.

    """

    C_TYPE = 'Model Predictive Control'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_range_max=mt.Async.C_RANGE_NONE,
                 p_state_thsld=0.00000001,
                 p_logging=Log.C_LOG_ALL):
        
        self.C_SCIREF_TYPE          = self.C_SCIREF_TYPE_ARTICLE
        self.C_SCIREF_AUTHOR        = "Grady Williams, Nolan Wagener, Brian Goldfain, Paul Drews, James M. Rehg, Byron Boots, Evangelos A. Theodorou"
        self.C_SCIREF_TITLE         = "Information theoretic MPC for model-based reinforcement learning"
        self.C_SCIREF_CONFERENCE    = "2017 IEEE International Conference on Robotics and Automation (ICRA)"
        self.C_SCIREF_YEAR          = "2017"
        self.C_SCIREF_MONTH         = "05"
        self.C_SCIREF_DOI           = "10.1109/ICRA.2017.7989202"
        
        ActionPlanner.__init__(self,
                                 p_state_thsld=p_state_thsld,
                                 p_logging=p_logging)
        mt.Async.__init__(self,
                          p_range_max=p_range_max, 
                          p_class_shared=mt.Shared,
                          p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def _plan_action(self, p_obs: State) -> SARSBuffer:
        """
        Custom planning algorithm to fill the internal action path (self._action_path). Search width
        and depth are restricted by the attributes self._width_limit and self._prediction_horizon.
        The default implementation utilizes MPC.

        Parameters
        ----------
        p_obs : State
            Observation data.

        Returns
        -------
        action_path : SARSBuffer
            Sequence of SARSElement objects with included actions that lead to the best possible reward.

        """
        
        if self._range == self.C_RANGE_NONE: 
            # initialize variable to store best path and its predicted overall reward
            best_path = None
            best_overall_reward = None
            
            for width in range(self._width_limit):
                state = p_obs
                path = SARSBuffer(p_size=self._prediction_horizon)
                overall_reward = 0
                
                for pred in range(self._prediction_horizon):
                    # generate random actions
                    action_values = np.zeros(self._envmodel._action_space.get_num_dim())
                    ids = self._envmodel._action_space.get_dim_ids()
                    for d in range(self._envmodel._action_space.get_num_dim()):
                        try:
                            base_set = self._envmodel._action_space.get_dim(ids[d]).get_base_set()
                        except:
                            raise ParamError('Mandatory base set is not defined.')
                            
                        try:
                            if len(self._envmodel._action_space.get_dim(ids[d]).get_boundaries()) == 1:
                                lower_boundaries = 0
                                upper_boundaries = self._envmodel._action_space.get_dim(ids[d]).get_boundaries()[0]
                            else:
                                lower_boundaries = self._envmodel._action_space.get_dim(ids[d]).get_boundaries()[0]
                                upper_boundaries = self._envmodel._action_space.get_dim(ids[d]).get_boundaries()[1]
                            if base_set == 'Z' or base_set == 'N':
                                action_values[d] = random.randint(lower_boundaries, upper_boundaries)
                            else:
                                action_values[d] = random.uniform(lower_boundaries, upper_boundaries)
                        except:
                            raise ParamError('Mandatory boundaries are not defined.')
                    action = Action(pred, self._envmodel._action_space, action_values)
                    
                    # compute next states and reward according to current state
                    next_state = self._envmodel.simulate_reaction(state, action)
                    reward = self._envmodel.compute_reward(p_state_old=state, p_state_new=next_state)
                    overall_reward += reward.get_overall_reward()
                    
                    # add to SARSBuffer
                    path.add_element(SARSElement(state, action, reward, next_state))
                    
                    # adjust the current state with next state
                    state = next_state
                
                # comparison between the current best path and the computed path
                if (best_path is None) or (overall_reward > best_overall_reward):
                    best_path = path
                    best_overall_reward = overall_reward
        else:
            best_path = self.execute(p_obs=p_obs)
                
        return best_path

    
## -------------------------------------------------------------------------------------------------
    def execute(self, **p_kwargs):
        for ids in range(self._width_limit):
            self._start_async(p_target=self._async_subtask,
                              p_tid=ids,
                              p_obs=p_kwargs['p_obs'])

        self.wait_async_tasks()
        
        result = self._so.get_results()
        result_list = list(zip(*result.values()))
        best_overall_reward = max(result_list[0])
        idx = result_list[0].index(best_overall_reward)
        
        return result_list[1][idx]


## -------------------------------------------------------------------------------------------------
    def _async_subtask(self, p_tid:int, p_obs:State):
        self._so.checkin(p_tid=p_tid)
        
        state = p_obs
        path = SARSBuffer(p_size=self._prediction_horizon)
        overall_reward = 0
        
        for pred in range(self._prediction_horizon):
            # generate random actions
            action_values = np.zeros(self._envmodel._action_space.get_num_dim())
            ids = self._envmodel._action_space.get_dim_ids()
            for d in range(self._envmodel._action_space.get_num_dim()):
                try:
                    base_set = self._envmodel._action_space.get_dim(ids[d]).get_base_set()
                except:
                    raise ParamError('Mandatory base set is not defined.')
                    
                try:
                    if len(self._envmodel._action_space.get_dim(ids[d]).get_boundaries()) == 1:
                        lower_boundaries = 0
                        upper_boundaries = self._envmodel._action_space.get_dim(ids[d]).get_boundaries()[0]
                    else:
                        lower_boundaries = self._envmodel._action_space.get_dim(ids[d]).get_boundaries()[0]
                        upper_boundaries = self._envmodel._action_space.get_dim(ids[d]).get_boundaries()[1]
                    if base_set == 'Z' or base_set == 'N':
                        action_values[d] = random.randint(lower_boundaries, upper_boundaries)
                    else:
                        action_values[d] = random.uniform(lower_boundaries, upper_boundaries)
                except:
                    raise ParamError('Mandatory boundaries are not defined.')
            action = Action(pred, self._envmodel._action_space, action_values)
            
            # compute next states and reward according to current state
            next_state = self._envmodel.simulate_reaction(state, action)
            reward = self._envmodel.compute_reward(p_state_old=state, p_state_new=next_state)
            overall_reward += reward.get_overall_reward()
            
            # add to SARSBuffer
            path.add_element(SARSElement(state, action, reward, next_state))
            
            # adjust the current state with next state
            state = next_state

        self._so.add_result( p_tid=p_tid, p_result=[overall_reward, path] )
        self._so.checkout( p_tid = p_tid )