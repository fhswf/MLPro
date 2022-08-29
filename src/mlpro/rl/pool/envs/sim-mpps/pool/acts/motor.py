## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.envs
## -- Module  : motor.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-24  0.0.0     SY/ML    Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2022-08-26)
This module provides a motor subsystem as default implementation.
To be noted, the usage of this simulation is not limited to RL tasks, but it also can be as a
testing environment for GT tasks, evolutionary algorithms, supervised learning, model predictive
control, and many more.
"""

from mpps import *

class Motor(Actuator):

    reg_m = []

    def __init__(self, p_name):
        Actuator.__init__(self)

        self.idx_m = len(self.reg_m)
        self.reg_m.append(self)
        self.name = p_name

        self.setup()

    def setup(self):

        print('hier werden transfer function initalisiert')
        # transport material function
        # energy consumption function

    def compute(self, **p_param):

        print('hier werden transfer function aufgerufen')

        return 0 