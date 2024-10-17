## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control.control_scenarios
## -- Module  : basic.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-04  0.1.0     DA       Initial implementation 
## -- 2024-10-09  0.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2024-10-09)

This module provides a simplified container class for a basic synchronous control system containing

- a controller
- a controlled system
- an optional integrator for the control variable

"""


from mlpro.bf.control.basics import ControlWorkflow
from mlpro.bf.various import Log
from mlpro.bf.control import Controller, ControlledSystem, ControlSystem
from mlpro.bf.control.operators import Comparator, Integrator



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ControlSystemBasic (ControlSystem):
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

    C_TYPE          = 'Control System Basic'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_controller : Controller,
                  p_controlled_system : ControlledSystem,
                  p_mode, 
                  p_ctrl_var_integration : bool = False,
                  p_cycle_limit = 0, 
                  p_visualize : bool = False, 
                  p_logging = Log.C_LOG_ALL ):
                
        self._controller           = p_controller
        self._controlled_system    = p_controlled_system
        self._ctrl_var_integration = p_ctrl_var_integration

        super().__init__( p_mode = p_mode,
                          p_cycle_limit = p_cycle_limit,
                          p_visualize = p_visualize,
                          p_logging = p_logging )
        

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize: bool, p_logging) -> ControlWorkflow:
        
        # 1 Create a new control cycle
        control_workflow = ControlWorkflow( p_mode = p_mode,
                                            p_visualize = p_visualize,
                                            p_logging = p_logging )
        

        # 2 Create and add a comparator
        comparator = Comparator( p_visualize = p_visualize, p_logging = p_logging )
        control_workflow.add_task( p_task = comparator )


        # 3 Add the controller
        control_workflow.add_task( p_task = self._controller, p_pred_tasks = [comparator] )


        # 4 Optionally create and add an integrator
        if self._ctrl_var_integration:
            integrator = Integrator( p_visualize = p_visualize, p_logging = p_logging )
            control_workflow.add_task( p_task = integrator, p_pred_tasks = [self._controller] )
            pred_sys = integrator

        else:
            pred_sys = self._controller


        # 5 Add the controlled system
        control_workflow.add_task( p_task = self._controlled_system, p_pred_tasks = [pred_sys] )
        self._controlled_system.system.set_mode( p_mode = p_mode )


        # 6 Initialize and return the prepared control workflow
        control_workflow.get_so().init( p_ctrlled_var_space = self._controlled_system.system.get_state_space(),
                                        p_ctrl_var_space = self._controlled_system.system.get_action_space() )
        
        return control_workflow