## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control
## -- Module  : howto_bf_control_PT_101_pid_controller_PT1.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-11-11  0.1.0     ASP       Creation
## -- 2024-12-03  0.2.0     ASP       Update
## -- 2025-01-06  0.3.0     ASP       Update HowTo
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2024-11-11)

This howto demonstrates a basic control system consisting of a PID
controller and a first-order-system (PT1) as a controlled system.

You will learn:

1) How to set up a basic control system with a PT1-controlled system and PID-Controller

2) How to execute a control system and to change the setpoint from outside

3) The Behaviour of the PT1 System in combination with a PID-Controller. Please vary with the system parameters K and T of the controlled System and
   the parameter Kp,Tn,Tv of the controller

"""

import numpy as np
from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.ops import Mode
from datetime import timedelta
from mlpro.bf.systems.pool import PT1
from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.bf.control.controlsystems import BasicControlSystem




# 1 Prepare for test
if __name__ == "__main__":
    logging     = Log.C_LOG_ALL
    visualize   = True
    step_rate   = 1
    num_dim     = 1
else:
    cycle_limit = 5
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 1
    num_dim     = 1


# 1.1 Define init parameters and calculate cycle limit

#init controlled systems parameter
pt1_K = 1
pt1_T = 5

# calculate cycle limit
cycle_time = pt1_T / (20 * pt1_K)
simulation_time = 3 * pt1_T
cycle_limit = int(simulation_time / cycle_time)

# init setpoint
setpoint_value = 10


# 2 Setup the control system

# 2.1 Controlled system
my_ctrl_sys = PT1(p_K = pt1_K,
                p_T = pt1_T,
                p_sys_num = 0,
                p_y_start = 0,
                p_latency = timedelta( seconds = cycle_time),
                p_visualize = visualize,
                p_logging = logging )

my_ctrl_sys.reset( p_seed = 1 )


# 2.2 Controller
my_ctrl = PIDController( p_input_space = my_ctrl_sys.get_state_space(),
                       p_output_space = my_ctrl_sys.get_action_space(),
                       p_Kp = 3.99,
                       p_Tn = 3.18,
                       p_Tv = 0.22,
                       p_integral_off = False,
                       p_derivitave_off = False,
                       p_name = 'PID Controller',
                       p_visualize = visualize,
                       p_logging = logging )


# 2.3 Basic control system
mycontrolsystem = BasicControlSystem( p_mode = Mode.C_MODE_SIM,
                                      p_controller = my_ctrl,
                                      p_controlled_system = my_ctrl_sys,
                                      p_name = 'First-Order-System and PID Controller',
                                      p_ctrl_var_integration = False,
                                      p_cycle_limit = cycle_limit,
                                      p_visualize = visualize,
                                      p_logging = logging )


# 3 Set initial setpoint values and reset the controlled system
mycontrolsystem.get_control_panels()[0][0].set_setpoint( p_values = np.ones(shape=(num_dim))*setpoint_value )
my_ctrl_sys.reset( p_seed = 1 )


# 4 Run some control cycles
if ( __name__ == '__main__' ) and visualize:
    mycontrolsystem.init_plot( p_plot_settings = PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                             p_view_autoselect = True,
                                                             p_step_rate = step_rate,
                                                             p_plot_horizon = 100 ) )
    input('\nPlease arrange all windows and press ENTER to start control processing...')

mycontrolsystem.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')