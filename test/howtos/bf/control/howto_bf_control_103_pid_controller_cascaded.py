## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : test/howtos/bf
## -- Module  : howto_bf_control_103_pid_controller_cascaded.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-01  0.0.0     DA       Creation
## -- 2024-10-31  0.1.0     ASP       Implementation Cascade Routine
## -- 2024-11-10  1.0.0     ASP       Refactor Cascade Routine
##                                      - Added HowTo description 
##                                      - Added PT2 Controlled System
##                                      - Changed paramters for the controlled system PT1
##                                      - Removed CasscadedSystem 
##                                      - adjust values of controlled systems properties and PID properties 
## -- 2024-12-03  1.1.0     ASP       Update PT1 und PT2  
## -- 2025-01-06  1.2.0     ASP       Update whole HowTo                         
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.0 (2025-01-06)

This HowTo is intended to simulate the temperature control of a stirred vessel. 
The cascade control consists of two cascades, the inner cascade consists of a P-controller and a first-order system, 
which represents the temperature behavior of the heating circuit. 
The outer cascade consists of a PID controller and a second-order system that determines the temperature behavior of the stirred vessel.  

You will learn:

1) How to set up a cascade control system with cascades consisting of a controller and a controlled system

2) How to execute a cascade control system and to change the setpoint from outside

3) Temperature behaviour of a stirred vessel

"""
from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.ops import Mode
import numpy as np
from datetime import timedelta
from mlpro.bf.systems.pool import PT1,PT2
from mlpro.bf.control.controlsystems import CascadeControlSystem
from mlpro.bf.control.controllers.pid_controller import PIDController



# 2 Prepare for test
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


# 2.1 Define init parameters and calculate cycle limit

#init controlled systems parameter
pt2_K = 1
pt2_D = 1.6165
pt2_w_0 = 0.00577
pt1_T = 1200
pt1_K = 25

# calculate cycle limit
simulation_time = 500 *1 / pt2_w_0
cycle_limit = int(simulation_time / 2)

# init setpoint
setpoint_value = 40


# 3 Setup inner casscade

# 3.1 controlled system 
my_ctrl_sys_1 = PT1(p_K = pt1_T,
                p_T = pt1_K,
                p_sys_num = 0,
                p_y_start = 0,
                p_latency = timedelta( seconds = 1 ),
                p_visualize = visualize,
                p_logging = logging )

my_ctrl_sys_1.reset( p_seed = 42 )   


# 3.2 P-Controller
my_ctrl_2 = PIDController( p_input_space = my_ctrl_sys_1.get_state_space(),
                    p_output_space = my_ctrl_sys_1.get_action_space(),
                    p_Kp = 0.36,
                    p_Tn = 0,
                    p_Tv = 0,
                    p_integral_off = True,
                    p_derivitave_off = True,
                    p_name = 'PID Controller2',
                    p_visualize = visualize,
                    p_logging = logging )


# 4 Setup outer casscade

# 4.1 controlled system 
my_ctrl_sys_2 = PT2(p_K = pt2_K,
                    p_D = pt2_D,
                    p_omega_0 = pt2_w_0,
                    p_sys_num = 1,
                    p_max_cycle = cycle_limit,
                    p_latency = timedelta( seconds = 4 ),
                    p_visualize = visualize,
                    p_logging = logging )

my_ctrl_sys_2.reset( p_seed = 42 )

#4.2 Init PID-Controller
my_ctrl_1 = PIDController( p_input_space = my_ctrl_sys_2.get_state_space(),
                    p_output_space = my_ctrl_sys_2.get_action_space(),
                    p_Kp = 9.43,
                    p_Tn = 228,
                    p_Tv = 50,
                    p_name = 'PID Controller',
                    p_visualize = visualize,
                    p_logging = logging )  


# 5. Cascaded control system
mycontrolsystem = CascadeControlSystem( p_mode = Mode.C_MODE_SIM,
                                        p_controllers = [ my_ctrl_1, my_ctrl_2],
                                        p_controlled_systems = [my_ctrl_sys_2, my_ctrl_sys_1 ],
                                        p_name = 'Stirring vessel',
                                        p_cycle_limit = cycle_limit,
                                        p_visualize = visualize,
                                        p_logging = logging )


# 6 Set initial setpoint values for all control workflows (=cascades) of the control system
for panel_entry in mycontrolsystem.get_control_panels():
    panel_entry[0].set_setpoint( p_values = np.ones(shape = (num_dim)) * setpoint_value )


# 7 Run some control cycles
if visualize:
    mycontrolsystem.init_plot( p_plot_settings = PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                            p_view_autoselect = True,
                                                            p_step_rate = step_rate,
                                                            p_plot_horizon = 100 ) )
input('\nPlease arrange all windows and press ENTER to start control processing...')    
    
mycontrolsystem.run()

if __name__ == '__main__':
    input('Press ENTER to exit...')

