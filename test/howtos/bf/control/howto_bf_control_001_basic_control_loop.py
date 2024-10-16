## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control
## -- Module  : howto_bf_control_001_basic_control_loop.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-11  0.0.0     DA       Creation
## -- 2024-10-09  0.1.0     DA       Refactoring
## -- 2024-10-13  0.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2024-10-13)

Let's play fox and hunter!

You will learn:

1) How to ...

2) How to ...

3) How to ...

"""

import numpy as np

from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.ops import Mode

from mlpro.bf.control import ControlledSystem
from mlpro.bf.control.controlsystems import ControlSystemBasic
from mlpro.bf.systems.pool import Fox
from mlpro.bf.control.controllers import Hunter






# 1 Preparation of demo/unit test mode
if __name__ == '__main__':
    # 1.1 Parameters for demo mode
    cycle_limit = 500
    num_dim     = 2
    logging     = Log.C_LOG_ALL
    visualize   = True
    step_rate   = 1
  
else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 5
    num_dim     = 2
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 1


# 2 Init control scenario

# 2.1 Control system
mycontrolledsystem = ControlledSystem( p_system = Fox( p_num_dim = num_dim ),
                                       p_name = 'Fox',
                                       p_visualize = visualize,
                                       p_logging = logging )

# 2.2 Controller
mycontroller = Hunter( p_input_space = mycontrolledsystem.system.get_state_space(),
                       p_output_space = mycontrolledsystem.system.get_action_space(),
                       p_name = 'Hunter',
                       p_visualize = visualize,
                       p_logging = logging )

# 2.3 Basic control system
mycontrolsystem = ControlSystemBasic( p_mode = Mode.C_MODE_SIM,
                                      p_controller = mycontroller,
                                      p_controlled_system = mycontrolledsystem,
                                      p_ctrl_var_integration = False,
                                      p_cycle_limit = cycle_limit,
                                      p_visualize = visualize,
                                      p_logging = logging )


# 3 Set initial setpoint values and reset the controlled system
mycontrolsystem.get_control_panel().set_setpoint( p_values = np.zeros(shape=(num_dim)) )
mycontrolledsystem.system.reset( p_seed = 1 )


# 4 Run some control cycles
if __name__ == '__main__':
    mycontrolsystem.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                             p_view_autoselect = False,
                                                             p_step_rate = step_rate,
                                                             p_plot_horizon = 100 ) )
    input('\nPlease arrange all windows and press ENTER to start control processing...')

mycontrolsystem.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')

