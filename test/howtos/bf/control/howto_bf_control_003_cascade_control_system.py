## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control
## -- Module  : howto_bf_control_003_cascade_control_system.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-11-09  1.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-11-09)

Let's play fox and hunter once again but this time as a cascade control system! This howto demonstrates 
a cascade control system consisting of two cascades with a dummy controller and a dummy controlled system. 
The dimensionality can be changed manually, where in 2d and 3d mode, a spatial visualization is carried out.

You will learn:

1) How to set up a cascade control system with cascades consisting of a controller and a controlled system

2) How to execute a cascade control system and to change the setpoint from outside

3) Different types of visualization depending on the dimensionality of the control system

"""

import numpy as np
from datetime import timedelta

from mlpro.bf import Log, PlotSettings, Mode

from mlpro.bf.systems.pool import Fox
from mlpro.bf.control.controllers import Hunter
from mlpro.bf.control.controlsystems import CascadeControlSystem




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

if __name__ == '__main__':
    i = input(f'\n\nDimensionality (press ENTER for {num_dim}): ')
    if i != '': num_dim = int(i)
    i = input(f'\nStep rate (visualization) (press ENTER for {step_rate}): ')
    if i != '': step_rate = int(i)



# 2 Setup the control system

# 2.1 Controller and controlled system of the outer cascade
my_ctrl_sys_1 = Fox( p_num_dim = num_dim,
                     p_latency = timedelta( seconds = 5 ),
                     p_visualize = visualize,
                     p_logging = logging )
my_ctrl_sys_1.reset( p_seed = 1 )

my_ctrl_1 = Hunter( p_input_space = my_ctrl_sys_1.get_state_space(),
                    p_output_space = my_ctrl_sys_1.get_action_space(),
                    p_name = 'Hunter1',
                    p_visualize = visualize,
                    p_logging = logging )


# 2.2 Controller and controlled system of the inner cascade
my_ctrl_sys_2 = Fox( p_num_dim = num_dim,
                     p_latency = timedelta( seconds = 1 ),
                     p_visualize = visualize,
                     p_logging = logging )
my_ctrl_sys_2.reset( p_seed = 2 )

my_ctrl_2 = Hunter( p_input_space = my_ctrl_sys_1.get_state_space(),
                    p_output_space = my_ctrl_sys_1.get_action_space(),
                    p_name = 'Hunter2',
                    p_visualize = visualize,
                    p_logging = logging )


# 2.3 Cascade control system
mycontrolsystem = CascadeControlSystem( p_mode = Mode.C_MODE_SIM,
                                        p_controllers = [ my_ctrl_1, my_ctrl_2 ],
                                        p_controlled_systems = [ my_ctrl_sys_1, my_ctrl_sys_2 ],
                                        p_name = 'Hunter and Fox',
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

