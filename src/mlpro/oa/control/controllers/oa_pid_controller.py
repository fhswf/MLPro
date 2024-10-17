## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.control.controllers
## -- Module  : oa_pid_controller.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-01  0.0.0     DA       Creation 
## -- 2024-09-26  0.0.0     ASP      Implementation RLPID, RLPIDOffPolicy 
## -- 2024-10-17  0.0.0     ASP      -Refactoring class RLPID
## --                                -change class name RLPIDOffPolicy to OffPolicyRLPID
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2024-09-01)

This module provides an implementation of a OA PID controller.

"""

from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.bf.ml.basics import *
from mlpro.rl import Policy,SARSElement
from mlpro.bf.control import ControlVariable, ControlledVariable




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLPID(Policy):
    """
    Policy class for closed loop control

    Parameters
    ----------
    p_pid_controller : PIDController,
        Instance of PIDController
    p_policy : Policy
        Policy algorithm
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                  p_observation_space: MSpace,
                  p_action_space: MSpace,
                  p_pid_controller:PIDController ,
                  p_policy:Policy,
                  p_id=None, 
                  p_buffer_size: int = 1, 
                  p_ada: bool = True, 
                  p_visualize: bool = False,
                  p_logging=Log.C_LOG_ALL ):
        
        super().__init__(p_observation_space, p_action_space, p_id, p_buffer_size, p_ada, p_visualize, p_logging)

        self._pid_controller = p_pid_controller
        self._policy = p_policy
        self._crtl_variable_old = None #None
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
        """
        Parameters:
        p_sars_elem:SARSElement
            Element of a SARSBuffer
        """

        is_adapted = False

        #get SARS Elements 
        p_state,p_crtl_variable,p_reward,p_state_new=tuple(p_sars_elem.get_data().values())

        
        if self._crtl_variable_old is not None:

           # create a new SARS
            p_sars_elem_new = SARSElement(p_state=p_state,
                                        p_action=self._crtl_variable_old,
                                        p_reward=p_reward, 
                                        p_state_new=p_state_new)
            
            #adapt own policy
            is_adapted = self._policy._adapt(p_sars_elem_new)        
                
            # compute new action with new error value (second s of Sars element)
            self._crtl_variable_old=self._policy.compute_action(p_obs=p_state_new)

            #get the pid paramter values 
            pid_values = self._crtl_variable_old.get_feature_data().get_values()

            #set paramter pid
            self._pid_controller.set_parameter(p_param={"Kp":pid_values[0],
                                                    "Ti":pid_values[1],
                                                    "Tv":pid_values[2]})
        else:
            #compute new action with new error value (second s of Sars element)
            self._crtl_variable_old = self._policy.compute_action(p_obs=p_state_new) 

        return is_adapted 
    

## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_obs: ControlledVariable) -> ControlVariable:  

        #get action 
        control_variable=self._pid_controller.compute_output(p_ctrl_error=p_obs)

        #return action
        return control_variable 
    




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OffPolicyRLPID(Policy):
    """
    OFF Policy class for closed loop control

    Parameters
    ----------
    p_pid_controller : PIDController,
        Instance of PIDController
    """

    def __init__(self, p_observation_space: MSpace, 
                 p_action_space: MSpace,
                 pid_controller:PIDController 
                 ,p_id=None, 
                 p_buffer_size: int = 1, 
                 p_ada: bool = True, 
                 p_visualize: bool = False, 
                 p_logging=Log.C_LOG_ALL ):
        
        super().__init__(p_observation_space, 
                         p_action_space, 
                         p_id, 
                         p_buffer_size, 
                         p_ada, p_visualize, p_logging)

        self._pid_controller = pid_controller 
        self._action_space = p_action_space.get_dim(p_id=0)


## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self, **p_par):

        # 1 add dim (Kp,Tn,Tv) in hp space 
        self._hyperparam_space.add_dim( self._action_space.get_dim(p_id=0))
        self._hyperparam_space.add_dim(self._action_space.get_dim(p_id=1))
        self._hyperparam_space.add_dim(self._action_space.get_dim(p_id=2))

        # # 2- create hp tuple from hp space 
        self._hyperparam_tuple = HyperParamTuple( p_set=self._hyperparam_space )

        # 3- set hp tuple values
        self._hyperparam_tuple.set_values(self._pid_controller.get_parameter_values())

    
## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_sars_elem: SARSElement) -> bool:
       return False
    
       
## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_obs: ControlledVariable) -> ControlVariable:  

        #compute control variable 
        control_variable=self._pid_controller.compute_output(p_ctrl_error=p_obs)

        return control_variable   

