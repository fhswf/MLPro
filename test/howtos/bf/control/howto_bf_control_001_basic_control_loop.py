## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control
## -- Module  : howto_bf_control_001_basic_control_loop.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-11  0.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2024-09-11)

This module demonstrates ...

You will learn:

1) How to ...

2) How to ...

3) How to ...

"""

from time import sleep

from mlpro.bf.control import Controller, ControlSystem, ControlCycle, ControlScenario
from mlpro.bf.systems.pool import DoublePendulumSystemS4




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyController (Controller):
    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyControlScenario (ControlScenario):

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_visualize: bool, p_logging):

        # 1 Prepare control system (e.g. by wrapping)
       ctrl_sys = ControlSystem( p_system = DoublePendulumSystemS4() )

        # 2 Prepare control cycle

        # ...





# 1 Preparation of demo/unit test mode
if __name__ == '__main__':
    # 1.1 Parameters for demo mode
    cycle_limit = 200
    logging     = Log.C_LOG_ALL
    visualize   = True
  
else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 2
    logging     = Log.C_LOG_NOTHING
    visualize   = False



# 2 Init control scenario
myscenario = MyControlScenario()
panel      = myscenario.get_control_panel()



# 3 Start the control process asynchronously
myscenario.run()
panel.start()



# 4 Change setpoint value in a loop
for i in range(10):
    sleep(1)
    panel.change_setpoint()



# 5 Stop control process
panel.stop()
    
