## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control
## -- Module  : howto_bf_control_PT_001_pid_control_PT1.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-12-03  0.1.0     AS       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2024-11-11)

The HowTo is intended to show the behavior of a first-order system in a closed loop control, without a controller

You will learn:

1) How to set up a PID control system with a PT1-controlled system

2) How to execute a control system and to change the setpoint from outside

3) The Behaviour of the PT1 System. Please vary with the system parameters K and T of the controlled System

"""

import numpy as np
from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.ops import Mode
from datetime import timedelta
from mlpro.bf.systems.pool import PT1
from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.bf.control.controlsystems import BasicControlSystem




# 1 Preparation of demo/unit test mode
pt1_T = 5
latency = 0.1 
cycle_limit = int(3*pt1_T/latency)
if __name__ == '__main__':
    # 1.1 Parameters for demo mode
    cycle_limit = cycle_limit
    num_dim     = 1
    logging     = Log.C_LOG_ALL
    visualize   = True
    step_rate   = 2
  
else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 5
    num_dim     = 4
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 1

if __name__ == '__main__':
    i = input(f'\n\nDimensionality (press ENTER for {num_dim}): ')
    if i != '': num_dim = int(i)
    i = input(f'\n\nStep rate (visualization) (press ENTER for {step_rate}): ')
    if i != '': step_rate = int(i)


setpoint_value =10

# 2 Setup the control system

# 2.1 Controlled system
my_ctrl_sys = PT1(p_K=5,
                p_T=pt1_T,
                p_sys_num=0,
                p_y_start=0,#setpoint_value,
                p_latency = timedelta( seconds =latency),
                p_visualize = visualize,
                p_logging = logging )

my_ctrl_sys.reset( p_seed = 1 )

# 2.2 Controller
#p_Kp = 1, p_integral_off = True, p_derivitave_off = True, it means PID Controller is not active and it forwards the control error 1 to 1
my_ctrl = PIDController( p_input_space = my_ctrl_sys.get_state_space(),
                       p_output_space = my_ctrl_sys.get_action_space(),
                       p_Kp=0.59,
                       p_Tn=1,
                       p_Tv=0.5,
                       p_integral_off=False,
                       p_derivitave_off=False,
                       p_name = 'PID Controller',
                       p_visualize = visualize,
                       p_logging = logging )

# 2.3 Basic control system
mycontrolsystem = BasicControlSystem( p_mode = Mode.C_MODE_SIM,
                                      p_controller = my_ctrl,
                                      p_controlled_system = my_ctrl_sys,
                                      p_name = 'First-Order-System',
                                      p_ctrl_var_integration = False,
                                      p_cycle_limit = cycle_limit,
                                      p_visualize = visualize,
                                      p_logging = logging )



# 3 Set initial setpoint values and reset the controlled system
mycontrolsystem.get_control_panels()[0][0].set_setpoint( p_values = np.ones(shape=(num_dim))*setpoint_value )
my_ctrl_sys.reset( p_seed = 1 )



# 4 Run some control cycles
if ( __name__ == '__main__' ) and visualize:
    mycontrolsystem.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                             p_view_autoselect = True,
                                                             p_step_rate = step_rate,
                                                             p_plot_horizon = 100 ) )
    input('\nPlease arrange all windows and press ENTER to start control processing...')

mycontrolsystem.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')

