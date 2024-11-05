## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : test/howtos/bf
## -- Module  : howto_bf_control_103_pid_controller_cascaded.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-01  0.0.0     DA       Creation
## -- 2024-10-31  0.1.0     AS       Implementation Cascade Routine
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2024-09-01)

This module demonstrates ...

You will learn:

1) How to ...

2) How to ...

3) How to ...

"""


from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.ops import Mode
from mlpro.bf.control import ControlledSystem,CascadedSystem
from mlpro.bf.control.controlsystems import ControlSystemBasic,CascadeControlSystem
from mlpro.bf.systems.pool import Fox,PT1
from mlpro.bf.control.controllers.pid_controller import PIDController
import gymnasium as gym
from mlpro_int_gymnasium.wrappers import WrEnvGYM2MLPro
import numpy as np



# 1 Preparation of demo/unit test mode
if __name__ == '__main__':
    # 1.1 Parameters for demo mode
    cycle_limit = 500
    num_dim     = 1
    logging     = Log.C_LOG_ALL
    visualize   = False
    step_rate   = 1
  
else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 5
    num_dim     = 1
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 1


# 2 Init control scenario
mysystem1 = PT1(p_K=25,p_T=1200,p_sys_num=0)
mysystem2 = PT1(p_K=25,p_T=1200,p_sys_num=1)
#2.1 Cascade System
mycascadedsystem1 = CascadedSystem( p_system = mysystem1,
                                   p_name = 'Cascade-system1',
                                   p_last_system=True,
                                   p_visualize = visualize,
                                   p_logging = logging )
#2.1 Cascade System
mycascadedsystem2 = CascadedSystem( p_system = mysystem2,
                                   p_name = 'Cascade-system2',
                                   p_last_system=False,
                                   p_visualize = visualize,
                                   p_logging = logging )

# 2.2 Controller
mycontroller = PIDController( p_input_space = mysystem1.get_state_space(),
                       p_output_space = mysystem1.get_action_space(),Kp=1.33,Ti=1.33,Tv=1.99,
                       p_name = 'PID Controller',
                       p_visualize = visualize,
                       p_logging = logging )
# 2.2 Controller
mycontroller2 = PIDController( p_input_space = mysystem2.get_state_space(),
                       p_output_space = mysystem2.get_action_space(),Kp=1.33,Ti=1.33,Tv=1.99,
                       p_name = 'PID Controller2',
                       p_visualize = visualize,
                       p_logging = logging )



mycontrolsystem=CascadeControlSystem( p_mode = Mode.C_MODE_SIM,
                                      p_controllers =[mycontroller,mycontroller2],
                                      p_cascaded_system= [mycascadedsystem1,mycascadedsystem2],
                                      p_ctrl_var_integration = False,
                                      p_cycle_limit = cycle_limit,
                                      p_visualize = visualize,
                                      p_logging = logging )


# 3 Set initial setpoint values and reset the controlled system
mycontrolsystem.get_control_panel().set_setpoint( p_values =[40])
mysystem1.reset( p_seed = 1 )
mysystem2.reset( p_seed = 1 )


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

