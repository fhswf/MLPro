## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_systems_001.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-19  0.0.0     LSB      Creation 
## -- 2023-04-19  1.0.0     LSB      Release 
## -- 2023-05-03  1.0.1     LSB      - Dark Mode
##                                   - Visuaization
## -------------------------------------------------------------------------------------------------

"""
This module illustrates the use of DemoScenario to demonstrate systems. This example runs all the 
native systems in MLPro for 10 cycles.

You will learn:

1. Setting up demo scenario

2. Running a system in MLPro

ver. 1.0.1 (2023-05-03)

"""

from mlpro.bf.systems import *
from mlpro.bf.systems.pool import DoublePendulumSystemS4, DoublePendulumSystemS7


# Checking for dark mode:
if __name__=='__main__':
    logging = Log.C_LOG_ALL
    visualize = True
    cycle_limit = 10
else:
    logging = Log.C_LOG_NOTHING
    visualize = False
    cycle_limit = 2


systems = [DoublePendulumSystemS4(p_logging=logging, p_visualize=visualize), DoublePendulumSystemS7(p_logging=logging, p_visualize=visualize)]


for system in systems:

    scenario = DemoScenario(p_system=system,
                            p_mode = Mode.C_MODE_SIM,
                            p_action_pattern = DemoScenario.C_ACTION_RANDOM,
                            p_cycle_limit=cycle_limit,
                            p_visualize=visualize,
                            p_logging=logging)
    
    scenario.run()