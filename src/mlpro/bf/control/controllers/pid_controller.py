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

from mlpro.bf.systems import Action
from mlpro.bf.control.basics import CTRLError, Controller




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PIDController (Controller):
    """
    PID controller.
    """

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
        
        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_ctrl_error: CTRLError) -> Action:
        """
        ...
        """
        
        raise NotImplementedError
