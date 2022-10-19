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
Ver. 0.0.0 (2022-10-18)

This module provides an example of using the transfer function method in MLPro for both default and
custom implementation.

You will learn:

1) How to use the default type of transfer function
    
2) How to set up your own function

"""


from mlpro.bf.math import *
from mlpro.rl.pool.envs.sim_mpps.mpps import TransferFunction
import math




if __name__ == "__main__":
    
    # 1. Using default type
    
    # 1.1. Initialize a given default transfer function
    myTF_linear = TransferFunction(p_name='Linear_TF',
                                   p_type=TransferFunction.C_TRF_FUNC_LINEAR,
                                   p_dt=0.01,
                                   m=5,
                                   b=2)
    
    
    # 1.2. Call the defined transfer function
    # 1.2.1. For a specific point
    p_input = 10
    output = myTF_linear.call(p_input)
    print(output)
    
    # 1.2.2. Within a specific range
    p_range = 5
    output = myTF_linear.call(p_input, p_range)
    print(output)
    
    # 1.3. Plot the graph
    myTF_linear.plot(p_input, p_input+p_range)
    
    
    # 2. Using own function
    
    # 2.1. Set up your custom function class
    class MyTransferFunction(TransferFunction):
        
        # 2.1.1. Set up which parameters required for your transfer function
        def set_function_parameters(self, p_args) -> bool:
            """
            y(t) = A cos(w * t - phi)
            """
            if self.get_type() == self.C_TRF_FUNC_CUSTOM:
                try:
                    self.A = p_args['A']
                    self.w = p_args['w']
                    self.phi = p_args['phi']
                except:
                    raise NotImplementedError('One/More parameters for this function is missing.')           
            return True
        
        # 2.1.2. Set up the mathematical calculation of your transfer function
        def custom_function(self, p_input, p_range=None):
            """
            y(t) = A cos(w * t - phi)
            """
            if p_range is None:
                return self.A * math.cos(self.w * p_input - self.phi)
            else:
                points = int(p_range/self.dt)
                output = 0
                for x in range(points+1):
                    current_input = p_input + x * self.dt
                    output += self.A * math.cos(self.w * current_input - self.phi)
                return output
                
    # 2.2. Initialize the transfer function
    myFunction = MyTransferFunction(p_name='DGL_solution',
                                    p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                    p_dt=0.05,
                                    A = 3.5,         # Current
                                    w = 314.15,      # angular velocity
                                    phi = -120)      # angle offset
    
    
    # 2.3. Call the defined transfer function
    # 2.3.1. For a specific point
    p_input = 0
    output = myFunction.call(p_input)
    print(output)
    
    # 2.3.2. Within a specific range
    p_range = 10
    output = myFunction.call(p_input, p_range)
    print(output)
    
    # 2.4. Plot the graph
    myFunction.plot(p_input, p_input+p_range)