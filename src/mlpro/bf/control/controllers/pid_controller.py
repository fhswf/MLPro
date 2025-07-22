## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control.controller
## -- Module  : pid_controller.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-01  0.0.0     DA       Creation 
## -- 2024-09-19  0.1.0     ASP      Implementation PIDController 
## -- 2024-10-17  0.2.0     ASP      Refactor PIDController 
## -- 2024-11-10  0.3.0     ASP      Refactor class PIDController: signature methode __init__() 
## -- 2024-11-16  0.4.0     ASP      Refactor class PIDController: signature methode __init__() 
## -- 2024-11-16  0.5.0     ASP      Changed Task.C_RANGE_NONE to Range.C_RANGE_NONE
## -- 2025-01-06  0.6.0     ASP      Refactor PIDController
## -- 2025-01-26  0.7.0     ASP      class PIDController: Changed parameters and attributes comments
## -- 2025-07-22  0.8.0     DA       Refactoring: __all__ export list, docstring, imports
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.8.0 (2025-07-22)

This module provides an implementation of a PID controller.

Learn more:

https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller

"""

import numpy as np 

from mlpro.bf.mt import Log,Range
from mlpro.bf.control import ControlError, Controller
from mlpro.bf.systems import ActionElement



# Export list for public API
__all__ = [ 'PIDController' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PIDController (Controller):

    """
    PID Controller for closed-loop controllers.

    Parameters
    ----------
    Kp                : float
        Gain factor 
    Tn                : float
        Settling time [s]
    Tv                : float
        Dead time [s]   
    p_integral_off    : Bool
        Disable integral term 
    p_derivitave_off  : Bool
        Disable derivitave term
    p_anti_windup_on  : float
        Enable anti windup filter 
    
    Attributes
    ----------
    _Kp              : float
        Gain factor 
    _Tn              : float
        settling time [s]
    _Tv              : float
        Dead time [s]   
    _integral_off    : Bool
        Disable integral term 
    _derivitave_off  : Bool
        Disable derivitave term
    _anti_windup_on  : float
        Enable anti windup filter 
    _integral_val    : float
        Overall integral value 
    _prev_error      : float 
        Last control error     
    _previous_tstamp : float 
        Last timestamp [s]
    _windup_limit    : list
        Value range of anti windup filter
    _output_limits   : list
        Value range of control variable
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_input_space,
                 p_output_space, 
                 p_Kp: float,
                 p_Tn: float = 0.0 ,
                 p_Tv: float= 0.0,
                 p_integral_off:bool = False,
                 p_derivitave_off:bool = False,
                 p_anti_windup_on: bool = False,
                 p_windup_limit:float = 0,
                 p_id = None, 
                 p_name = None,                  
                 p_range_max=Range.C_RANGE_NONE, 
                 p_visualize = False, 
                 p_logging = Log.C_LOG_ALL, 
                 **p_kwargs):
        
        super().__init__(p_input_space, 
                         p_output_space, 
                         p_id, 
                         p_name, 
                         p_range_max, 
                         p_visualize, 
                         p_logging, 
                         **p_kwargs)
        

        self._Kp = p_Kp
        self._Tn = p_Tn  
        self._Tv = p_Tv  
        self._integral_off = p_integral_off
        self._derivitave_off = p_derivitave_off
        self._anti_windup_on = p_anti_windup_on
        self._integral_val = 0.0
        self._prev_error = 0.0      
        self._previous_tstamp = None 
        self._windup_limit = p_windup_limit
        self._output_limits = p_output_space.get_dims()[0].get_boundaries()
   

## -------------------------------------------------------------------------------------------------
    def set_parameter(self, **p_param):
        """
        Sets/changes the parameters of the PID controller.

        Parameters
        ----------
        Kp : float
            gain factor 
        Tn : float
            settling time 
        Tv : float
            dead time
        """

        # set kp value
        self._Kp = p_param['p_param']['Kp']
        # set Ti value
        self._Tn = p_param['p_param']['Tn']
        #set Tv value
        self._Tv =p_param['p_param']['Tv']


## -------------------------------------------------------------------------------------------------
    def get_parameter_values(self)-> np.ndarray:
        return np.array([self._Kp,self._Tn,self._Tv])
    

## -------------------------------------------------------------------------------------------------        
    def _compute_output(self, p_ctrl_error: ControlError, p_ctrl_var: ActionElement):

        """
        Custom method to compute and an action based on an incoming control error. The result needs
        to be stored in the action element handed over. I/O values can be accessed as follows:

        SISO
        ----
        Get single error value: control_error_siso = p_ctrl_error.values[0]
        Set single action value: p_action_element.values[p_ae_id] = action_siso


        Parameters
        ----------
        p_ctrl_error : CTRLError
            Control error.
        p_ctrl_var : ActionElement
            Control element to be filled with resulting control value(s).
        """

        #get control error
        control_error_siso = p_ctrl_error.values[0]

        #time delta
        dt = 0

        #get the time stamp 
        tstamp = p_ctrl_error.get_tstamp()

        #calculate time difference
        if self._previous_tstamp is not None:
            dt = tstamp- self._previous_tstamp
            dt = dt.total_seconds()

        #propertional term 
        p_term = self._Kp * control_error_siso

        #integral term
        i_term = 0

        #ignore i term , if it is disabled
        if not self._integral_off:

            #calculat integral term
            self._integral_val += control_error_siso*dt

            # anti - windup 
            if self._anti_windup_on and self._windup_limit is not None:
                self._integral_val = max(min(self._integral_val, self._windup_limit), - self._windup_limit)

            #calculate i term , if Ti not zero
            if self._Tn != 0:
                 i_term = (self._Kp / self._Tn) * self._integral_val 

        # derivitave term 
        d_term = 0

        #ignore i term , if it is disabled or delta is equal zero 
        if dt > 0 and not self._derivitave_off:
            d_term = self._Kp * self._Tv * (control_error_siso - self._prev_error) / dt
        
        #compute control variable value 
        control_variable_siso = p_term + i_term + d_term

        #apply control variable limits
        lower_bound, upper_bound = tuple(self._output_limits)

        control_variable_siso = min(max(control_variable_siso,lower_bound), upper_bound)

        # safe the current values for the next iteration
        self._prev_error = control_error_siso
        self._previous_tstamp = tstamp        

        #set control value
        p_ctrl_var._set_values([control_variable_siso])