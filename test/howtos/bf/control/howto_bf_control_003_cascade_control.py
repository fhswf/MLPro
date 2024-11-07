## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control
## -- Module  : howto_bf_control_003_cascade_control.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-11-07  0.1.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2024-11-07)

Let's play fox and hunter, but now as a cascade control!

You will learn:

1) How to set up a cascade control

2) How to ...

3) How to ...

"""

import numpy as np

from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.ops import Mode

from mlpro.bf.control import *
from mlpro.bf.control.operators import Comparator, Converter
from mlpro.bf.systems.pool import Fox
from mlpro.bf.control.controllers import Hunter




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CascadeControlSystem (ControlSystem):
    """
    ...
    """
    
## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize, p_logging):

        # 1 Setup inner control loop
        workflow_inner = ControlWorkflow( p_mode = p_mode,
                                          p_name = 'Inner',
                                          p_visualize = p_visualize,
                                          p_logging = p_logging )
        
        # 1.1 Create the inner controlled system
        t_ctrl_sys = ControlledSystem( p_system = Fox( p_visualize = p_visualize, p_logging = p_logging ),
                                       p_visualize = p_visualize,
                                       p_logging = p_logging )
        
        t_ctrl_sys.system.reset( p_seed = 1)
        

        # 1.2 Create and add a comparator
        t_comp = Comparator( p_visualize = p_visualize,
                             p_logging = p_logging )
        
        workflow_inner.add_task( p_task = t_comp )


        # 1.3 Create and add a controller
        t_ctrl = Hunter( p_input_space = t_ctrl_sys.system.get_state_space(),
                         p_output_space = t_ctrl_sys.system.get_action_space(),
                         p_name = 'Hunter #1',
                         p_visualize = p_visualize,
                         p_logging = p_logging )
        
        workflow_inner.add_task( p_task = t_ctrl, p_pred_tasks = [ t_comp ] )
        

        # 1.4 Add the controlled system
        workflow_inner.add_task( p_task = t_ctrl_sys, p_pred_tasks = [ t_ctrl ] )


        # 2 Setup and return outer control loop
        workflow_outer = ControlWorkflow( p_mode = p_mode,
                                          p_name = 'Outer',
                                          p_visualize = p_visualize,
                                          p_logging = p_logging )
        

        # 2.1 Create the outer controlled system
        t_ctrl_sys = ControlledSystem( p_system = Fox( p_visualize = p_visualize, p_logging = p_logging ),
                                       p_visualize = p_visualize,
                                       p_logging = p_logging )
        
        t_ctrl_sys.system.reset( p_seed = 2 )
        

        # 2.2 Create and add a comparator
        t_comp = Comparator( p_visualize = p_visualize,
                             p_logging = p_logging )
        
        workflow_outer.add_task( p_task = t_comp )


        # 2.3 Create and add a controller
        t_ctrl = Hunter( p_input_space = t_ctrl_sys.system.get_state_space(),
                         p_output_space = t_ctrl_sys.system.get_action_space(),
                         p_name = 'Hunter #2',
                         p_visualize = p_visualize,
                         p_logging = p_logging )
        
        workflow_outer.add_task( p_task = t_ctrl, p_pred_tasks = [ t_comp ] )


        # 2.4 Create and add a converter ( control variable -> setpoint)
        t_conv = Converter( p_src_type = ControlVariable,
                            p_dst_type = SetPoint,
                            p_visualize = p_visualize,
                            p_logging = p_logging )
        
        workflow_outer.add_task( p_task = t_conv, p_pred_tasks = [ t_ctrl ] )


        # 2.5 Add the inner control workflow as task
        workflow_outer.add_task( p_task = workflow_inner, p_pred_tasks = [ t_conv ] )


        # 2.6 Create and add a converter ( controlled variable -> control variable )
        t_conv = Converter( p_src_type = ControlledVariable,
                            p_dst_type = ControlVariable,
                            p_visualize = p_visualize,
                            p_logging = p_logging )
        
        workflow_outer.add_task( p_task = t_conv, p_pred_tasks = [ workflow_inner ] )

        return workflow_outer




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


# 2 Init cascade control system
mycontrolsystem = CascadeControlSystem( p_mode = Mode.C_MODE_SIM,
                                         p_cycle_limit = cycle_limit,
                                         p_visualize = visualize,
                                         p_logging = logging )


# 3 Set initial setpoint
mycontrolsystem.get_control_panel().set_setpoint( p_values = np.zeros(shape=(num_dim)) )


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

