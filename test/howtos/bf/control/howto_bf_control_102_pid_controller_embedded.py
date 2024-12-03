## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : test/howtos/bf
## -- Module  : howto_bf_control_102_pid_controller_embedded.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-01  0.0.0     DA       Creation
## -- 2024-10-28  0.1.0     DA       Implementation PID CartPole Control
## -- 2024-10-28  0.2.0     ASP      Update PID CartPole Control

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
from mlpro.bf.control import ControlledSystem
from mlpro.bf.control.controllers.pid_controller import PIDController
import gymnasium as gym
from mlpro_int_gymnasium.wrappers import WrEnvGYM2MLPro
from mlpro.bf.control.controlsystems import BasicControlSystem
import numpy as np







# 1 Preparation of demo/unit test mode
if __name__ == '__main__':
    # 1.1 Parameters for demo mode
    cycle_limit = 500
    num_dim     = 1
    logging     = Log.C_LOG_ALL
    visualize   = True
    step_rate   = 1
  
else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 5
    num_dim     = 1
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 1


if visualize:
    gym_env = gym.make('CartPole-v1', render_mode="human")
else:
    gym_env = gym.make('CartPole-v1')

# 2.1 Control system
mycontrolledsystem = ControlledSystem( p_system =  WrEnvGYM2MLPro( p_gym_env=gym_env, p_visualize=visualize, p_logging=logging),
                                       p_name = 'Cart Pole',
                                       p_visualize = visualize,
                                       p_logging = logging )
mycontrolledsystem.system.reset()

# 2.2 Controller
mycontroller = PIDController( p_input_space = mycontrolledsystem.system.get_state_space(),
                       p_output_space = mycontrolledsystem.system.get_action_space(),p_Kp=1.33,p_Ti=1.33,p_Tv=1.99,
                       p_name = 'PID Controller',
                       p_visualize = visualize,
                       p_logging = logging )



# 2.3 Basic control system
mycontrolsystem = BasicControlSystem( p_mode = Mode.C_MODE_SIM,
                                      p_controller = mycontroller,
                                      p_controlled_system = mycontrolledsystem,
                                      p_name = 'PID+Cartpole',
                                      p_ctrl_var_integration = False,
                                      p_cycle_limit = cycle_limit,
                                      p_visualize = visualize,
                                      p_logging = logging )



# 3 Set initial setpoint values and reset the controlled system
mycontrolsystem.get_control_panels()[0][0].set_setpoint( p_values = np.zeros(shape=(num_dim)) )
mycontrolledsystem.reset( p_seed = 1 )


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
