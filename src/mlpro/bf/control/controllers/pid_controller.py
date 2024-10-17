## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control.controller
## -- Module  : pid_controller.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-01  0.0.0     DA       Creation 
## -- 2024-09-19  0.0.0     ASP      Implementation PIDController 
## -- 2024-10-17  0.0.0     ASP      Refactor PIDController 
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2024-09-01)

This module provides an implementation of a PID controller.

Learn more:

https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller

"""

from mlpro.bf.math.basics import MSpace
from mlpro.bf.mt import Log, Task
from mlpro.bf.control.basics import ControlError, Controller
from mlpro.bf.systems.basics import ActionElement
import numpy as np 




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PIDController (Controller):

    """
    PID Controller for closed-loop controllers.

    Parameters
    ----------
    Kp : float
        gain factor 
    Ti : float
        settling time [s]
    Tv : float
        dead time [s]   
    disable_integral: Bool
        disable integral term 
    disable_derivitave: Bool
        disable derivitave term
    enable_anti_windup: float
        enable anti windup filter 

    ...
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,Kp: float,  
                 p_input_space: MSpace, 
                 p_output_space: MSpace, 
                 Ti: float = 0.0 ,
                 Tv: float= 0.0,
                 disable_integral:bool = False,
                 disable_derivitave:bool = False,
                 enable_anti_windup: bool = False,
                 windup_limit:float =0,
                 p_id=None, p_name: str = None,
                 p_range_max=Task.C_RANGE_NONE,
                 p_visualize: bool = False, 
                 p_logging=Log.C_LOG_ALL, 
                 **p_kwargs):
        
        super().__init__(p_input_space, 
                         p_output_space, 
                         p_id, p_name, p_range_max, 
                         p_visualize, p_logging, 
                         **p_kwargs)

        self.Kp = Kp
        self.Ti = Ti  
        self.Tv = Tv  
        self.disable_integral = disable_integral
        self.disable_derivitave = disable_derivitave
        self.integral = 0.0
        self.prev_error = 0.0      
        self.previous_tstamp = None 
        self.enable_windup = enable_anti_windup
        self.windup_limit = windup_limit
        self.output_limits = p_output_space.get_dims()[0].get_boundaries()


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
        self.Kp = p_param['p_param']['Kp']
        # set Ti value
        self.Ti = p_param['p_param']['Ti']
        #set Tv value
        self.Tv =p_param['p_param']['Tv']


## -------------------------------------------------------------------------------------------------
    def get_parameter_values(self)-> np.ndarray:
        return np.array([self.Kp,self.Ti,self.Tv])
    

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
        control_error_siso = p_ctrl_error.get_values()[0]

        #time delta
        dt = 0

        #get the time stamp 
        tstamp = p_ctrl_error.get_tstamp()

        #calculate time difference
        if self.previous_tstamp is not None:
            dt = tstamp- self.previous_tstamp
            dt = dt.total_seconds()

        #propertional term 
        p_term = self.Kp * control_error_siso

        #integral term
        i_term = 0

        #ignore i term , if it is disabled
        if not self.disable_integral:

            #calculat integral term
            self.integral += control_error_siso*dt

            # anti - windup 
            if self.enable_windup and self.windup_limit is not None:
                self.integral = max(min(self.integral, self.windup_limit), -self.windup_limit)

            #calculate i term , if Ti not zero
            if self.Ti != 0:
                 i_term = (self.Kp/self.Ti)* self.integral 

        # derivitave term 
        d_term =0

        #ignore i term , if it is disabled or delta is equal zero 
        if dt> 0 and not self.disable_derivitave:
            d_term = self.Kp*self.Tv*(control_error_siso- self.prev_error)/dt
        
        #compute control variable value 
        control_variable_siso = p_term+i_term+d_term

        #apply control variable limits
        lower_bound, upper_bound = tuple(self.output_limits)

        control_variable_siso = min(max(control_variable_siso,lower_bound), upper_bound)

        # safe the current values for the next iteration
        self.prev_error = control_error_siso
        self.previous_tstamp = tstamp        

        #set control value
        p_ctrl_var._set_values([control_variable_siso])
        











        

        



        



       


        
        
        
    

