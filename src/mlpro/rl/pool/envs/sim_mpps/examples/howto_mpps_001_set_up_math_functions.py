## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.envs.sim_mpps
## -- Module  : mpps.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-24  0.0.0     SY/ML    Creation
## -- 2022-??-??  1.0.0     SY/ML    Release of first version
## -------------------------------------------------------------------------------------------------


"""
Ver. 0.0.0 (2022-09-15)

This module provides different math functions and his applcation

"""

from mlpro.bf.math import *
from mpps import TransferFunction

import math

# create the function space
math_function = MSpace()

# initalisation of the math function
# each function depens on the name, id and fix parameters
# Function has as basic math function linear, sinus and cosinus
math_function.add_dim(TransferFunction(p_name='linear', p_id=None, arg0=5, arg1=3))
math_function.add_dim(TransferFunction(p_name="cosinus",  p_id=None, arg0=0.8))

# the math function space listed all functions with ids
math_ids = math_function.get_dim_ids()

# ids can be assign
LINEAR = math_ids[0]
COSINE = math_ids[1]

# functions can be called for calculation
linear_volume = math_function.get_dim(LINEAR).call(5)


# Function can be extends by own math function
class TransferFunction(TransferFunction):
    """
    Extends class Function by own DGL
    """
    def DGL_solution(self, *p_value):
        # y(t) = A cos(w * t - phi)
        return self.args["arg0"] * math.cos(self.args["arg1"] * p_value[0] - self.args["arg2"])


# now the function can be add to the math space (Note: Function name must be simular)
A = 3.5         # Current
w = 314.15      # angular velocity
phi = -120      # angle offset

math_function.add_dim(TransferFunction(p_name="DGL_solution",  p_id=None, arg0=A, arg1=w, arg2=phi))

# ids must be called again
math_ids = math_function.get_dim_ids()

# get math_id
my_DGL_solution = math_ids[2]

# compute current after 15 secents
current = math_function.get_dim(my_DGL_solution).call(15)