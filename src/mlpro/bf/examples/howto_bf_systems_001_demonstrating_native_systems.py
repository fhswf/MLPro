## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_systems_001.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-19  0.0.0     LSB      Creation 
## -- 2023-04-19  1.0.0     LSB      Release 
## -------------------------------------------------------------------------------------------------

"""
This module illustrates the use of DemoScenario to demonstrate systems. This example runs all the 
native systems in MLPro for 10 cycles.

You will learn:

1. Setting up demo scenario

2. Running a system in MLPro

ver. 1.0.0 (2023-04-19)

"""

from mlpro.bf.systems import *
from mlpro.bf.systems.pool import DoublePendulumSystemS4, DoublePendulumSystemS7


# Checking for dark mode:
if __name__=='__main__':
    p_logging = Log.C_LOG_ALL
else:
    p_logging = Log.C_LOG_NOTHING




systems = [DoublePendulumSystemS4(), DoublePendulumSystemS7()]


for system in systems:

    scenario = DemoScenario(p_system=system,
                            p_mode = Mode.C_MODE_SIM,
                            p_action_pattern = DemoScenario.C_ACTION_RANDOM,
                            p_cycle_limit=10,
                            p_visualize=False,
                            p_logging=p_logging)
    
    scenario.run()