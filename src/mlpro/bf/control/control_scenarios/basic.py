## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control.control_scenarios
## -- Module  : basic.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-04  0.1.0     DA       Initial implementation 
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2024-10-04)

This module provides a simplified container class for a basic synchronous control scenario containing

- a controller
- a control system
- an optional action incrementer

"""


from mlpro.bf.control.basics import ControlCycle
from mlpro.bf.various import Log
from mlpro.bf.control import Controller, ControlSystem, ControlScenario
from mlpro.bf.control.operators import Comparator, Cumulator



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlScenarioBasic (ControlScenario):
    """
    Simplified container class for a basic synchronous control scenario containing

    - a controller
    - a control system
    - an optional action incrementer

    Parameters
    ----------
    p_controller : Controller
        Controller to be used in the control loop
    p_control_system : ControlSystem
        Control system to be used in the control loop
    p_cumulated_action : bool = False
        If True, an optional action cumulator is added to control loop
    """

    C_TYPE          = 'Control Cycle Basic'
    C_NAME          = ''

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_controller : Controller,
                  p_control_system : ControlSystem,
                  p_mode, 
                  p_cumulated_actions : bool = False,
                  p_cycle_limit=0, 
                  p_visualize:bool=False, 
                  p_logging=Log.C_LOG_ALL ):
                
        self._controller          = p_controller
        self._control_system      = p_control_system
        self._cumulated_actions   = p_cumulated_actions

        super().__init__( p_mode = p_mode,
                          p_cycle_limit = p_cycle_limit,
                          p_visualize = p_visualize,
                          p_logging = p_logging )
        

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize: bool, p_logging) -> ControlCycle:
        
        # 1 Create a new control cycle
        control_cycle = ControlCycle( p_visualize = p_visualize,
                                      p_logging = p_logging )
        

        # 2 Create and add a comparator
        comparator = Comparator( p_visualize = p_visualize,
                                 p_logging = p_logging )
        
        control_cycle.add_task( p_task = comparator )


        # 3 Add the controller
        control_cycle.add_task( p_task = self._controller, p_pred_tasks = [comparator] )

        # 4 Optionally add a cumulator
        if self._cumulated_actions:
            cumulator = Cumulator( p_visualize = p_visualize,
                                  p_logging = p_logging )
            
            control_cycle.add_task( p_task = cumulator, p_pred_tasks = [self._controller] )
            pred_sys = cumulator

        else:
            pred_sys = self._controller
        
        
        

