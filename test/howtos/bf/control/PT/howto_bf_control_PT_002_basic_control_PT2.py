## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control
## -- Module  : howto_bf_control_PT_002_basic_control_PT2.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-11-10  0.1.0     AS       Creation
## -- 2024-12-03  0.2.0     AS       Update PT2 system
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2024-12-93)

The HowTo is intended to show the behavior of a second-order system in a closed loop control, without a controller.
Further infos about relevant beahvious: https://www.circuitbread.com/tutorials/second-order-systems-2-3

You will learn:

1) How to set up a basic control system with a PT1-controlled system

2) How to execute a control system and to change the setpoint from outside

3) The Behaviour of the PT2 System. Please vary with the system parameters K,D and omega_0 of the controlled System

"""

import numpy as np
from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.ops import Mode
from datetime import timedelta
from mlpro.bf.systems.pool import PT2
from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.bf.control.controlsystems import BasicControlSystem




# 1 Preparation of demo/unit test mode
if __name__ == '__main__':
    # 1.1 Parameters for demo mode
    cycle_limit = 100
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
my_ctrl_sys = PT2(p_K=1,
                    p_D=1,
                    p_omega_0=5,
                    p_sys_num=1,                    
                    p_max_cycle=cycle_limit,
                    p_y_start =0,
                    p_latency = timedelta( seconds = 0.01 ),
                    p_visualize = visualize,
                    p_logging = logging )



my_ctrl_sys.reset( p_seed = 1 )

# 2.2 Controller
#p_Kp = 1, p_integral_off = True, p_derivitave_off = True, it means PID Controller is not active and it forwards the control error 1 to 1
my_ctrl = PIDController( p_input_space = my_ctrl_sys.get_state_space(),
                       p_output_space = my_ctrl_sys.get_action_space(),
                       p_Kp=1,
                       p_Tn=0,
                       p_Tv=0,
                       p_integral_off=True,
                       p_derivitave_off=True,
                       p_name = 'PID Controller',
                       p_visualize = visualize,
                       p_logging = logging )

# 2.3 Basic control system
mycontrolsystem = BasicControlSystem( p_mode = Mode.C_MODE_SIM,
                                      p_controller = my_ctrl,
                                      p_controlled_system = my_ctrl_sys,
                                      p_name = 'Second-Order-System',
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

