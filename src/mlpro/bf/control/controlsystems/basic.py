## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control.control_scenarios
## -- Module  : basic.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-04  0.1.0     DA       Initial implementation 
## -- 2024-10-09  0.2.0     DA       Refactoring
## -- 2024-11-09  0.3.0     DA       Refactoring
## -- 2025-07-18  0.4.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.4.0 (2025-07-18)

This module provides a simplified container class for a basic synchronous control system containing

- a controller
- a controlled system
- an optional integrator for the control variable

"""

from typing import Union

from mlpro.bf import Log
from mlpro.bf.systems import System
from mlpro.bf.control import Controller, ControlledSystem
from mlpro.bf.control.controlsystems import CascadeControlSystem
from mlpro.bf.control.operators import Integrator



# Export list for public API
__all__ = [ 'BasicControlSystem' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BasicControlSystem (CascadeControlSystem):
    """
    Simplified container class for a basic synchronous control system containing

    - a controller
    - a controlled system
    - an optional integrator for the control variable

    Parameters
    ----------
    p_mode
        Operation mode. See Mode.C_VALID_MODES for valid values. Default = Mode.C_MODE_SIM.
    p_controller : Controller
        Controller to be used in the control workflow
    p_controlled_system : ControlledSystem
        Controlled system to be used in the control workflow
    p_name : str = ''
        Name of the control system
    p_cycle_limit : int
        Maximum number of cycles. Default = 0 (no limit).
    p_ctrl_var_integration : bool = False
        If True, an optional intrator is added to control workflow
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.  
    p_kwargs : dict
        Custom keyword parameters handed over to custom method setup().
    """

    C_TYPE          = 'Basic Control System'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_mode, 
                  p_controller : Controller,
                  p_controlled_system : Union[System, ControlledSystem],
                  p_ctrl_var_integration : bool = False,
                  p_name : str = '',
                  p_cycle_limit = 0, 
                  p_visualize : bool = False, 
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):
        
        if p_ctrl_var_integration:
            controllers= [ [ p_controller, 
                             Integrator( p_range_max = p_controller.get_range(),
                                         p_visualize = p_visualize,
                                         p_logging = p_logging ) ] ]
        else:
            controllers = [ p_controller ]

        
        super().__init__( p_mode = p_mode,
                          p_controllers = controllers,
                          p_controlled_systems = [ p_controlled_system ],
                          p_name = p_name,
                          p_cycle_limit = p_cycle_limit,
                          p_visualize = p_visualize,
                          p_logging = p_logging,
                          **p_kwargs )
