## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control
## -- Module  : howto_bf_control_001_basic_control_loop.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-11  0.0.0     DA       Creation
## -- 2024-10-09  0.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2024-10-09)

This module demonstrates ...

You will learn:

1) How to ...

2) How to ...

3) How to ...

"""

from time import sleep

from mlpro.bf.control.basics import ControlError
from mlpro.bf.systems.basics import ActionElement
from mlpro.bf.various import Log
from mlpro.bf.ops import Mode
from mlpro.bf.control import Controller, ControlledSystem, ControlSystemBasic
from mlpro.bf.systems.pool import DoublePendulumSystemS4




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyController (Controller):
    
    def _compute_output(self, p_ctrl_error: ControlError, p_ctrl_var_elem: ActionElement):
        pass





# 1 Preparation of demo/unit test mode
if __name__ == '__main__':
    # 1.1 Parameters for demo mode
    cycle_limit = 1
    logging     = Log.C_LOG_ALL
    visualize   = False
  
else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 5
    logging     = Log.C_LOG_NOTHING
    visualize   = False


# 2 Init control scenario

# 2.1 Control system
mycontrolledsystem = ControlledSystem( p_system = DoublePendulumSystemS4() )

# 2.2 Controller
mycontroller = MyController( p_input_space = mycontrolledsystem.system.get_state_space(),
                             p_output_space = mycontrolledsystem.system.get_action_space(),
                             p_id = 0,
                             p_visualize = visualize,
                             p_logging = logging )

# 2.3 Basic control system
mycontrolsystem = ControlSystemBasic( p_mode = Mode.C_MODE_SIM,
                                      p_controller = mycontroller,
                                      p_controlled_system = mycontrolledsystem,
                                      p_cycle_limit = cycle_limit,
                                      p_visualize = visualize,
                                      p_logging = logging )


# 3 Set initial setpoint values
mycontrolsystem.get_control_panel().set_setpoint( p_values = [0,0,0,0])


# 4 Start the control process
mycontrolsystem.run()