## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control
## -- Module  : howto_bf_control_001_basic_control_loop.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-11  0.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2024-09-11)

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
from mlpro.bf.control import Controller, ControlSystem, ControlScenarioBasic
from mlpro.bf.systems.pool import DoublePendulumSystemS4




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyController (Controller):
    
    def _compute_action(self, p_ctrl_error: ControlError, p_action_element: ActionElement):
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
mycontrolsystem = ControlSystem( p_system = DoublePendulumSystemS4() )

# 2.2 Controller
mycontroller = MyController( p_error_space = mycontrolsystem.system.get_state_space(),
                             p_action_space = mycontrolsystem.system.get_action_space(),
                             p_id = 0,
                             p_visualize = visualize,
                             p_logging = logging )

                             
# 2.3 Basic control scenario
myscenario = ControlScenarioBasic( p_mode = Mode.C_MODE_SIM,
                                   p_controller = mycontroller,
                                   p_control_system = mycontrolsystem,
                                   p_cycle_limit = cycle_limit,
                                   p_visualize = visualize,
                                   p_logging = logging )


# 3 Set initial setpoint values
myscenario.get_control_panel().set_setpoint( p_values = [0,0,0,0])


# 4 Start the control process
myscenario.run()