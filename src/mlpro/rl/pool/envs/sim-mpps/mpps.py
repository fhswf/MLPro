## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.envs
## -- Module  : mpps.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-24  0.0.0     SY/ML    Creation
## -- 2022-??-??  1.0.0     SY/ML    Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2022-08-24)

This module provides a multi-purpose environment of a continuous and batch production systems with
modular settings and high-flexibility.

The users are able to develop and simulate their own production systems including setting up own
actuators, reservoirs, modules/stations, production sequences and many more. We also provide the
default implementations of actuators, reservoirs, and modules, which can be found in the pool of
objects.

To be noted, the usage of this simulation is not limited to RL tasks, but it also can be as a
testing environment for GT tasks, evolutionary algorithms, supervised learning, model predictive
control, and many more.
"""


from mlpro.rl.models import *
from mlpro.bf.various import *
import numpy as np
import random




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
