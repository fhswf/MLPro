## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.control.control_scenarios
## -- Module  : basic.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-07  0.1.0     DA       Initial implementation 
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2024-10-07)

This module provides a simplified container class for a basic synchronous oa control scenario containing

- a controller
- an oa control system
- an optional action cumulator

"""


from mlpro.bf.various import Log
from mlpro.bf.control import ControlPanel, ControlWorkflow, ControlSystemBasic
from mlpro.oa.control import OAControlSystem



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAControlSystemBasic: # (OAControlSystem, ControlSystemBasic):
    """
    Simplified container class for a basic synchronous control system containing

    - a controller
    - an oa control system
    - an optional action incrementer
    """

    C_TYPE          = 'OA Control Cycle Basic'
    C_NAME          = ''

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize: bool, p_logging) -> ControlWorkflow:

        control_workflow = ControlSystemBasic._setup(self, p_mode, p_visualize, p_logging)

        try:
            control_workflow.get_control_panel().register_event_handler( p_event_id = ControlPanel.C_EVENT_ID_SETPOINT_CHG,
                                                                      p_event_handler = self._controller.hdl_setpoint_changed )
        except:
            self.log(Log.C_LOG_TYPE_W, 'Controller is not online adaptive')

        return control_workflow