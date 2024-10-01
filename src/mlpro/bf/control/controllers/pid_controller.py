## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control.controller
## -- Module  : pid_controller.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-01  0.0.0     DA       Creation 
## -- 2024-09-19  0.0.0     ASP      Implemebtation PIDController 
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2024-09-01)

This module provides an implementation of a PID controller.

Learn more:

https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller

"""

from mlpro.bf.math.basics import Log,Set
from mlpro.bf.mt import Log, Task
from mlpro.bf.systems import Action
from mlpro.bf.control.basics import ControlError, Controller
from mlpro.bf.systems.basics import ActionElement
from mlpro.bf.various import Log
from datetime import datetime, timedelta
import numpy as np 




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PIDController (Controller):
    """
    PID controller.
    """


    def __init__(self,Kp: float, Ti: float = 0.0 ,Tv: float= 0.0,disable_integral:bool = False,disable_derivitave:bool = False,enable_windup: bool = False,windup_limit:float =0,output_limits: tuple = (0,100) ,p_name: str = None, p_range_max=Task.C_RANGE_THREAD, p_duplicate_data: bool = False, p_visualize: bool = False, p_logging=Log.C_LOG_ALL, **p_kwargs):
        super().__init__(p_name, p_range_max, p_duplicate_data, p_visualize, p_logging, **p_kwargs)

        self.Kp = Kp
        self.Ti = Ti  # [s] 
        self.Tv = Tv  # [s]
        self.disable_integral = disable_integral
        self.disable_derivitave = disable_derivitave
        self.integral = 0.0
        self.prev_error = 0.0      
        self.previous_time = None #[datetime]
        self.enable_windup = enable_windup
        self.windup_limit = windup_limit
        self.output_limits = output_limits

## -------------------------------------------------------------------------------------------------
    def set_parameter(self, **p_param):
        """
        Sets/changes the parameters of the PID controller.

        Parameters
        ----------
        Kp : float
            gain factor 
        Ti : float
            settling time 
        Tv : float
            dead time
        """

        # set kp value
        self.Kp = p_param.get('Kp',self.Kp)
        # set Ti value
        self.Ti = p_param.get('Ti',self.Ti)
        #set Tv value
        p_param.get('Tv',self.Tv)     
## -------------------------------------------------------------------------------------------------
    def get_parameter_values(self)-> np.ndarray:
        return np.array([self.Kp,self.Ti,self.Tv])

## -------------------------------------------------------------------------------------------------

    def _compute_action(self, p_ctrl_error: ControlError, p_action_element: ActionElement, p_ctrl_id: int = 0, p_ae_id: int = 0):  

        """
        Custom method to compute and an action based on an incoming control error. The result needs
        to be stored in the action element handed over. I/O values can be accessed as follows:

        SISO
        ----
        Get single error value: error_siso = p_ctrl_error.values[p_ctrl_id]
        Set single action value: p_action_element.values[p_ae_id] = action_siso

        MIMO
        ----
        Get multiple error values: error_mimo = p_ctrl_error.values
        Set multiplie action values: p_action_element.values = action_mimo


        Parameters
        ----------
        p_ctrl_error : CTRLError
            Control error.
        p_action_element : ActionElement
            Action element to be filled with resulting action value(s).
        p_ctrl_id : int = 0
            SISO controllers only. Id of the related source value in p_ctrl_error.
        p_ae_id : int = 0 
            SISO controller olny. Id of the related destination value in p_action_element.
        """


        #get control error
        error_siso = p_ctrl_error.get_feature_data().get_values()[p_ctrl_id]

        delta_time = 0

        current_time = p_ctrl_error.get_tstamp()

        #calculate time difference
        if self.previous_time is not None:
            delta_time = current_time- self.previous_time
            delta_time = delta_time.total_seconds()

        #propertional term 
        p_term = self.Kp * error_siso

        #integral term
        i_term = 0

        #ignore i term , if it is disabled
        if not self.disable_integral:

            #calculat integral term
            self.integral += error_siso*delta_time

            # anti - windup 
            if self.enable_windup and self.windup_limit is not None:
                self.integral = max(min(self.integral, self.windup_limit), -self.windup_limit)

            #calculate i term , if Ti not zero
            if self.Ti != 0:
                 i_term = (self.Kp/self.Ti)* self.integral 


        # derivitave term 
        d_term =0

        #ignore i term , if it is disabled or delta is equal zero 
        if delta_time> 0 and not self.disable_derivitave:
            d_term = self.Kp*self.Tv*(error_siso- self.prev_error)/delta_time
        
        #compute action value 
        action_siso = p_term+i_term+d_term

        #apply action limits
        lower_bound, upper_bound = self.output_limits

        action_siso = min(max(action_siso,lower_bound), upper_bound)

        # safe the current values for the next iteration
        self.prev_error = error_siso
        self.previous_time = current_time        

        #set action value
        p_action_element.set_values([action_siso])
        











        

        



        



       


        
        
        
    

