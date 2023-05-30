## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.systems.pool
## -- Module  : doublependulum.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-mm-mm  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-dd-mm)

This module provides the online adaptive extensions of the Double Pendulum System.

"""


from mlpro.oa.systems import *
from mlpro.bf.ml.systems.pool.doublependulum import *
from mlpro.bf.systems.pool.doublependulum import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DoublePendulumOA4(DoublePendulumA4, OASystem):

    C_NAME = 'DoublePendulumOA4'





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DoublePendulumOA7(DoublePendulumA7, DoublePendulumOA4):

    C_NAME = 'DoublePendulumOA7'



