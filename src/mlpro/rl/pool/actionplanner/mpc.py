## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.pool.actionplanner
## -- Module  : mpc
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-15  0.0.0     SY       Creation
## -- 2022-08-15  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-08-15)

This module provides a default implementation of model predictive control (MPC).
"""

from mlpro.rl.models import *
import random
         
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MPC (ActionPlanner, ScientificObject):
    """
    Template class for MPC to be used as part of model-based planning agents. 
    The goal is to find the best sequence of actions that leads to a maximum reward.

    Parameters
    ----------
    p_state_thsld : float
        Threshold for metric difference between two states to be equal. Default = 0.00000001.
    p_logging
        Log level (see constants of class Log). Default = Log.C_LOG_ALL.

    """

    C_TYPE = 'Model Predictive Control'


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

        self.C_SCIREF_TYPE          = self.C_SCIREF_TYPE_ARTICLE
        self.C_SCIREF_AUTHOR        = "Grady Williams, Nolan Wagener, Brian Goldfain, Paul Drews, James M. Rehg, Byron Boots, Evangelos A. Theodorou"
        self.C_SCIREF_TITLE         = "Information theoretic MPC for model-based reinforcement learning"
        self.C_SCIREF_CONFERENCE    = "2017 IEEE International Conference on Robotics and Automation (ICRA)"
        self.C_SCIREF_YEAR          = "2017"
        self.C_SCIREF_MONTH         = "05"
        self.C_SCIREF_DOI           = "10.1109/ICRA.2017.7989202"
        
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
                        if base_set == 'Z' and base_set == 'N':
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
                
        return best_path