## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : test/howtos/bf
## -- Module  : howto_bf_control_103_pid_controller_cascaded.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-01  0.0.0     DA       Creation
## -- 2024-10-31  0.1.0     AS       Implementation Cascade Routine
## -- 2024-11-10  1.0.0     AS       Refactor Cascade Routine
##                                      - Added HowTo description 
##                                      - Added PT2 Controlled System
##                                      - Changed paramters for the controlled system PT1
##                                      - Removed CasscadedSystem 
##                                      - adjust values of controlled systems properties and PID properties                     
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-11-10)

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



# 1 Preparation of demo/unit test mode
if __name__ == '__main__':
    # 1.1 Parameters for demo mode
    cycle_limit = 500
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



# 2 Setup the control system
# 2.1 Controller and controlled system of the outer cascade
my_ctrl_sys_1 = PT1(p_K=25,
                p_T=1200,
                p_sys_num=0,
                p_latency = timedelta( seconds = 5 ),
                p_visualize = visualize,
                p_logging = logging )

my_ctrl_sys_1.reset( p_seed = 1 )

# 2.2 Controller
my_ctrl_1 = PIDController( p_input_space = my_ctrl_sys_1.get_state_space(),
                       p_output_space = my_ctrl_sys_1.get_action_space(),
                       p_Kp=9.43,
                       p_Tn=228,
                       p_Tv=50,
                       p_name = 'PID Controller',
                       p_visualize = visualize,
                       p_logging = logging )


# 2.2 Controller and controlled system of the inner cascade
my_ctrl_sys_2 = PT2(K=1,
                    p_D=1.6165,
                    p_omega_0=0.00577,
                    p_sys_num=1,
                    p_max_cycle=cycle_limit,
                    p_latency = timedelta( seconds = 5 ),
                    p_visualize = visualize,
                    p_logging = logging )

my_ctrl_sys_2.reset( p_seed = 2 )

# 2.2 P-Controller
my_ctrl_2 = PIDController( p_input_space = my_ctrl_sys_2.get_state_space(),
                       p_output_space = my_ctrl_sys_2.get_action_space(),
                       p_Kp=0.36,
                       p_Tn=0,
                       p_Tv=0,
                       p_integral_off=True,
                       p_derivitave_off=True,
                       p_name = 'PID Controller2',
                       p_visualize = visualize,
                       p_logging = logging )


# 2.3 Cascade control system
mycontrolsystem = CascadeControlSystem( p_mode = Mode.C_MODE_SIM,
                                        p_controllers = [ my_ctrl_1, my_ctrl_2 ],
                                        p_controlled_systems = [ my_ctrl_sys_1, my_ctrl_sys_2 ],
                                        p_name = 'Stirring vessel',
                                        p_cycle_limit = cycle_limit,
                                        p_visualize = visualize,
                                        p_logging = logging )



# 3 Set initial setpoint values for all control workflows (=cascades) of the control system
for panel_entry in mycontrolsystem.get_control_panels():
    panel_entry[0].set_setpoint( p_values = np.zeros(shape=(num_dim)) )



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

