## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.control.controllers
## -- Module  : oa_pid_controller.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-01  0.0.0     DA       Creation 
## -- 2024-09-26  0.1.0     ASP      Implementation RLPID, RLPIDOffPolicy 
## -- 2024-10-17  0.2.0     ASP      -Refactoring class RLPID
## --                                -change class name RLPIDOffPolicy to OffPolicyRLPID
## -- 2024-11-10  0.3.0     ASP      -Removed class OffPolicyRLPID
## -- 2024-12-05  0.4.0     ASP      -Add plot methods
## -- 2024-12-05  0.5.0     ASP      -changed signature of compute_action()
## -- 2024-12-05  0.6.0     ASP      -implementation assign_so(), update compute_action()
## -- 2024-12-06  0.7.0     ASP      -BugFix: _adapt()
## -- 2025-01-02  0.8.0     ASP      -Renaming of variable names 
## -- 2025-01-26  0.9.0     ASP      class RLPID: Changed parameters and attributes comments
## -- 2025-06-11  1.0.0     DA       Refactoring
## -- 2025-07-22  1.1.0     DA       Refactoring: __all__ export list, docstring, imports
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-22)

This module provides an implementation of a OA PID controller.

"""

from mlpro.bf import Log
from mlpro.bf.mt import Shared
from mlpro.bf.math import MSpace
from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.bf.streams import InstDict
from mlpro.bf.systems import Action, State
from mlpro.bf.ml import *

from mlpro.rl import SARSElement,Policy



# Export list for public API
__all__ = [ 'RLPID' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLPID(Policy):
    """
    Policy class for closed loop control

    Parameters
    ----------
    p_observation_space : MSpace
        Observation space of the RLPID
    p_action_space      : MSpace
        Action space of the RLPID
    p_pid_controller    : PIDController,
        Instance of PIDController
    p_policy            : Policy
        Policy algorithm

    
    Attributes
    ----------
    _pid_controller : PIDController
        Internal PID-Controller
    _policy         : Policy
        Policy algorithm
    _action_old     : Action
        Last action
    _action_space   : MSpace
        Action space of RLPID 
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                  p_observation_space: MSpace,
                  p_action_space: MSpace,
                  p_pid_controller:PIDController ,
                  p_policy:Policy,
                  p_id = None, 
                  p_buffer_size: int = 1, 
                  p_ada: bool = True, 
                  p_visualize: bool = False,
                  p_logging=Log.C_LOG_ALL ):
        
        super().__init__(p_observation_space,
                        p_action_space, p_id,
                        p_buffer_size, 
                        p_ada, p_visualize,
                        p_logging)

        self._pid_controller = p_pid_controller
        self._policy = p_policy
        self._action_old = None 
        self._action_space = p_action_space


## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self, **p_par):

        # 1 Create a dispatcher hyperparameter tuple for the RLPID policy
        self._hyperparam_tuple = HyperParamDispatcher(p_set=self._hyperparam_space)

        # 2 Extend RLPID policy's hp space and tuple from policy
        try:
            self._hyperparam_space.append( self._policy.get_hyperparam().get_related_set(), p_new_dim_ids=False)
            self._hyperparam_tuple.add_hp_tuple(self._policy.get_hyperparam())
        except:
            pass


## -------------------------------------------------------------------------------------------------
    def get_hyperparam(self) -> HyperParamTuple:
       return self._policy.get_hyperparam()
    
    
## -------------------------------------------------------------------------------------------------
    def _update_hyperparameters(self) -> bool:
       return self._policy._update_hyperparameters()  
      

## -------------------------------------------------------------------------------------------------    
    def _adapt(self, p_sars_elem: SARSElement) -> bool:

        is_adapted = False

        #get SARS Elements 
        p_state,p_crtl_variable,p_reward,p_state_new=tuple(p_sars_elem.get_data().values())
        
        if self._action_old is not None:

           # create a new SARS
            p_sars_elem_new = SARSElement(p_state = p_state,
                                        p_action = self._action_old,
                                        p_reward = p_reward, 
                                        p_state_new = p_state_new)       


            #adapt own policy
            is_adapted = self._policy._adapt(p_sars_elem_new)

            if is_adapted:     
                
                # compute new action with new error value (second s of Sars element)
                self._action_old=self._policy.compute_action(p_obs=p_state_new)

                #get the pid paramter values 
                pid_values = self._action_old.get_feature_data().get_values()

                #set paramter pid
                self._pid_controller.set_parameter(p_param={"Kp":pid_values[0],
                                                    "Tn":pid_values[1],
                                                    "Tv":pid_values[2]})         
            
        else:

            #compute new action with new error value (second s of Sars element)
            self._action_old = self._policy.compute_action(p_obs=p_state_new) 

        return is_adapted 
    

## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_obs: State) -> Action:  

        #get action 
        control_variable=self._pid_controller.compute_output(p_ctrl_error=p_obs)

        #return action
        return Action(p_action_space=control_variable.get_feature_data().get_related_set(),
               p_values=control_variable.values, 
               p_tstamp=control_variable.tstamp)       
    
    
## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure, p_settings):
        return self._pid_controller._init_plot_2d(p_figure, p_settings)
    

## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure, p_settings):
        return self._pid_controller._init_plot_3d(p_figure, p_settings)
    

## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure, p_settings):
        return self._pid_controller._init_plot_nd(p_figure, p_settings)
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings, p_instances, **p_kwargs) -> bool:
        return self._pid_controller._update_plot_2d(p_settings = p_settings, p_instances = p_instances, **p_kwargs)
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings, p_instances, **p_kwargs) -> bool:
        return self._pid_controller._update_plot_3d(p_settings = p_settings, p_instances = p_instances, **p_kwargs)
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings, p_instances:InstDict, **p_kwargs) -> bool: 
        return self._pid_controller._update_plot_nd(p_settings = p_settings, p_instances = p_instances, **p_kwargs)   


## -------------------------------------------------------------------------------------------------
    def _remove_plot_2d(self):
        return self._pid_controller._remove_plot_2d()
    

## -------------------------------------------------------------------------------------------------
    def _remove_plot_3d(self):
        return self._pid_controller._remove_plot_3d()
    

## -------------------------------------------------------------------------------------------------
    def _remove_plot_nd(self):
        return self._pid_controller._remove_plot_nd()


## -------------------------------------------------------------------------------------------------
    def assign_so(self, p_so:Shared):
        """
        Assigns an existing shared object to the task. The task takes over the range of asynchronicity
        of the shared object if it is less than the current one of the task.

        Parameters
        ----------
        p_so : Shared
            Shared object.
        """

        self._pid_controller.assign_so(p_so=p_so)

