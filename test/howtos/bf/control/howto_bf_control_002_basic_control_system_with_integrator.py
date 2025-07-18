## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control
## -- Module  : howto_bf_control_002_basic_control_system_with_integrator.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-11-09  1.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-11-09)

Let's play fox and hunter once again! This howto demonstrates a basic control system consisting of a dummy
controller and a dummy controlled system. Additionally an integrator is placed between controller and 
controlled system. The dimensionality can be changed manually, where in 2d and 3d mode, a spatial 
visualization is carried out.

You will learn:

1) How to set up a basic control system with a controller, a controlled system, and an integrator

2) How to execute a control system and to change the setpoint from outside

3) Different types of visualization depending on the dimensionality of the control system

"""


import numpy as np

from mlpro.bf import Log, PlotSettings, Mode

from mlpro.bf.systems.pool import Fox
from mlpro.bf.control.controllers import Hunter
from mlpro.bf.control.controlsystems import BasicControlSystem




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
    i = input(f'\n\nStep rate (visualization) (press ENTER for {step_rate}): ')
    if i != '': step_rate = int(i)



# 2 Setup the control system

# 2.1 Controlled system
mycontrolledsystem = Fox( p_num_dim = num_dim,
                          p_visualize = visualize,
                          p_logging = logging )

# 2.2 Controller
mycontroller = Hunter( p_input_space = mycontrolledsystem.get_state_space(),
                       p_output_space = mycontrolledsystem.get_action_space(),
                       p_name = 'Hunter',
                       p_visualize = visualize,
                       p_logging = logging )

# 2.3 Basic control system
mycontrolsystem = BasicControlSystem( p_mode = Mode.C_MODE_SIM,
                                      p_controller = mycontroller,
                                      p_controlled_system = mycontrolledsystem,
                                      p_name = 'Hunter and Fox (+I)',
                                      p_ctrl_var_integration = True,
                                      p_cycle_limit = cycle_limit,
                                      p_visualize = visualize,
                                      p_logging = logging )



# 3 Set initial setpoint values and reset the controlled system
mycontrolsystem.get_control_panels()[0][0].set_setpoint( p_values = np.zeros(shape=(num_dim)) )
mycontrolledsystem.reset( p_seed = 1 )



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

