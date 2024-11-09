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
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.3.0 (2024-11-09)

This module provides a simplified container class for a basic synchronous control system containing

- a controller
- a controlled system
- an optional integrator for the control variable

"""


from mlpro.bf.various import Log

from mlpro.bf.control import Controller, ControlledSystem
from mlpro.bf.control.controlsystems import CascadeControlSystem
from mlpro.bf.control.operators import Integrator




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
    p_controller : Controller
        Controller to be used in the control workflow
    p_controlled_system : ControlledSystem
        Controlled system to be used in the control workflow
    p_ctrl_var_integration : bool = False
        If True, an optional intrator is added to control workflow
    """

    C_TYPE          = 'Basic Control System'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_controller : Controller,
                  p_controlled_system : ControlledSystem,
                  p_mode, 
                  p_ctrl_var_integration : bool = False,
                  p_cycle_limit = 0, 
                  p_visualize : bool = False, 
                  p_logging = Log.C_LOG_ALL ):
        
        controllers = [ p_controller ]

        if p_ctrl_var_integration:
            controllers.append( Integrator( p_range_max = p_controller.get_range(),
                                            p_visualize = p_visualize,
                                            p_logging = p_logging ) )
        
        super().__init__( p_controllers = controllers,
                          p_controlled_systems = [ p_controlled_system ],
                          p_mode = p_mode,
                          p_cycle_limit = p_cycle_limit,
                          p_visualize = p_visualize,
                          p_logging = p_logging )