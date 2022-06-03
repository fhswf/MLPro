## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto_bf_002_timer.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-05-19  1.0.0     DA       Creation
## -- 2021-09-11  1.0.0     MRD      Change Header information to match our new library name
## -- 2021-11-13  1.0.1     DA       Minor fix
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2021-11-13)

This module demonstrates the Timer class functionality.
"""


import time
import random
from datetime import timedelta
from mlpro.bf.various import Timer, Log




# Demo class
class TimerDemo (Log):

    C_TYPE  = 'Demo class'
    C_NAME  = 'Timer'

    def __init__(self, p_timer:Timer):
        self.timer = p_timer
        super().__init__()


    def log(self, p_type, *p_args):
        super().log(p_type, 'Process time', self.timer.get_time(), ', Cycle', self.timer.get_lap_id(), 'Lap time', self.timer.get_lap_time(), '--', *p_args)


    def run_step(self, p_step_id):
        self.log(self.C_LOG_TYPE_I, 'Process step', p_step_id, 'started')
        duration = 0.6 * random.random()
        time.sleep(duration)
        self.log(self.C_LOG_TYPE_I, 'Process step', p_step_id, 'ended after', duration, 'seconds')


    def run_cycle(self):
        self.run_step(1) 
        self.run_step(2) 
        self.run_step(3) 
        if not self.timer.finish_lap():
            self.log(self.C_LOG_TYPE_W, 'Last process cycle timed out!!')


    def run(self):
        for i in range(10):
            self.run_cycle()




if __name__ == "__main__":

    # Example 1
    print('\n\n\nExample 1: Timer in virtual time mode with lap duration 1 day and 15 seconds...\n\n')
    t = Timer(Timer.C_MODE_VIRTUAL, timedelta(1,15,0))
    d = TimerDemo(t)
    d.run()


    # Example 2
    print('\n\n\nExample 2: Timer in real time mode with lap duration 1 second and forced timeout situations...\n\n')
    t = Timer(Timer.C_MODE_REAL, timedelta(0,1,0))
    d = TimerDemo(t)
    d.run()
