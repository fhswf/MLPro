## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_zz_999_description.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-01  0.0.0     FN       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-01-01)

This module demonstrates ...

You will learn:

1) How to implement a P, PI and PID Controller

2) How to use a PID Controller without and with anti wind up mechanism

3) How to use a controller to create a control signal and set new pid parameters

"""


from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.bf.control.basics import CTRLError, Controller
from mlpro.bf.math.basics import Log
from mlpro.bf.mt import Log, Task
from mlpro.bf.various import Log
from mlpro.bf.math import *
from mlpro.bf.systems import Action, ActionElement



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyPIDController (PIDController):
    """
    This class demonstrates how to ...
    """

    # needed for proper logging (see class mlpro.bf.various.Log)
    C_TYPE      = 'Demo'
    C_NAME      = 'PID Control'

## -------------------------------------------------------------------------------------------------
    def __init__(self, Kp: float, Ti: float, Tv: float, disable_integral: bool = False, disable_derivitave: bool = False, enable_windup: bool = False, windup_limit: float = 0, output_limits: tuple = ..., p_name: str = None, p_range_max=Task.C_RANGE_THREAD, p_duplicate_data: bool = False, p_visualize: bool = False, p_logging=Log.C_LOG_ALL, **p_kwargs):
        super().__init__(Kp, Ti, Tv, disable_integral, disable_derivitave, enable_windup, windup_limit, output_limits, p_name, p_range_max, p_duplicate_data, p_visualize, p_logging, **p_kwargs)


# 1 Preparation of demo test
if __name__ == '__main__':


    # 1.Parameters for demo mode
    Kp = 12 
    Ti = 10
    Tv = 2.5
    element = Element(Set())
    print(element.get_dim_ids())
    element.set_values([12])
    crtl_error = CTRLError(element,p_tstamp=datetime.now())

    action_element= ActionElement(Set())


    # 2. create diffrent instances of the class pid_controller
    # create an instance of a  P-Controller
    p_controller = PIDController(p_Kp=Kp,p_integral_off=True, p_derivitave_off=True)
    # create an instance of a PI-Controller
    pi_controller = PIDController(p_Kp=True, p_Tn=10,p_derivitave_off=True)
    # create an instance of a  PID-Controller without anti wind up mechanism
    pid_controller = PIDController(p_Kp=True, p_Tn=10,p_Tv=Tv)
    # create an instance of a  PID-Controller with anti wind up mechanism
    pid_controller_antiWindUp = PIDController(p_Kp=True, p_Tn=10,p_Tv=Tv,p_anti_windup_on=True,p_windup_limit=100,output_limits=(0,100))


    # 3. run compute action 
    p_controller._compute_action(crtl_error,action_element)
    p_controller._compute_action(crtl_error,action_element)
    pi_controller._compute_action(crtl_error,action_element)
    pid_controller._compute_action(crtl_error,action_element)
    pid_controller_antiWindUp._compute_action(crtl_error,action_element)


    # 4. set new pid parameter 
    p_controller.set_parameter(Kp= 6)
    pi_controller.set_parameter(Kp= 6,Ti=4)
    pid_controller.set_parameter(Kp=6,Ti=4,Tv=12)


                                                               
                     
  

