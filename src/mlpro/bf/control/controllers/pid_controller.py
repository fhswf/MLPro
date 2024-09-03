## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control.controller
## -- Module  : pid_controller.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-01  0.0.0     DA       Creation 
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
from mlpro.bf.control.basics import CTRLError, Controller
from mlpro.bf.various import Log
from datetime import datetime, timedelta
import numpy as np 




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PIDController (Controller):
    """
    PID controller.
    """


    def __init__(self,Kp: float, Ti: float,Td: float,disable_integral:bool = False,disable_derivitave:bool = False,enable_windup: bool = False,windup_limit:float =0,output_limits: tuple = (0,100) ,p_name: str = None, p_range_max=Task.C_RANGE_THREAD, p_duplicate_data: bool = False, p_visualize: bool = False, p_logging=Log.C_LOG_ALL, **p_kwargs):
        super().__init__(p_name, p_range_max, p_duplicate_data, p_visualize, p_logging, **p_kwargs)

        self.Kp = Kp
        self.Ti = Ti  # [s] 
        self.Td = Td  # [s]
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
        p_par1 : type1
            Description 1
        p_par2 : type2
            Description 2
        p_par3 : type3
            Description 3
        """

        # set kp value
        self.Kp = p_param['Kp']
        # set ki value
        self.Ti = p_param['Ti']
        #set kd value
        self.Td = p_param['Td']      

    

## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_ctrl_error: CTRLError) -> Action:

        #get control error
        crtl_error = p_ctrl_error.get_feature_data().get_values()[0]
        delta_time = 0
        current_time =datetime.now()

        #calculate time difference
        if self.previous_time is not None:
            delta_time = current_time- self.previous_time
            delta_time = delta_time.total_seconds()

        #propertional term 
        p_term = self.Kp * crtl_error

        #integral term
        i_term = 0

        #ignore i term , if it is disabled
        if not self.disable_integral:

            #calculat integral term
            self.integral += crtl_error*delta_time

            # anti - windup 
            if self.enable_windup and self.windup_limit is not None:
                self._integral = max(min(self._integral, self.windup_limit), -self.windup_limit)

            #calculate i term , if Ti not zero
            if self.Ti != 0:
                 i_term = (self.Kp/self.Ti)* self.integral 


        # derivitave term 
        d_term =0

        #ignore i term , if it is disabled or delta is equal zero 
        if delta_time> 0 and not self.disable_derivitave:
            d_term = self.Kp*self.Td*(crtl_error- self.prev_error)/delta_time
        
        #compute action value 
        output = p_term+i_term+d_term

        #apply action limits
        lower_bound, upper_bound = self.output_limits

        output = min(max(output,lower_bound), upper_bound)

        # safe the current values for the next iteration
        self.prev_error = crtl_error
        self.previous_time = current_time
        output = np.array([output],dtype=float)

        #return action value
        return Action(p_action_space=Set(),p_values=[output])




        



       


        
        
        
    

