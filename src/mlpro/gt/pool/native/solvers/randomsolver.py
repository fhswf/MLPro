## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.pool.native.solvers
## -- Module  : randomsolver.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-09-21  0.0.0     SY       Creation
## -- 2023-09-22  1.0.0     SY       Release of first version
## -- 2024-01-12  1.0.1     SY       Refactoring: Module Name
## -- 2025-07-18  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-18) 

This module provides solver with random GT strategy.
"""

import random

import numpy as np

from mlpro.bf import ParamError
from mlpro.gt.native.basics import *
         
        

# Export list for public API
__all__ = [ 'RandomSolver' ]  




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RandomSolver (GTSolver):
    """
    A solver that generates random actions for each dimension of the underlying strategy space.
    """

    C_NAME      = 'RandomSolver'


## -------------------------------------------------------------------------------------------------
    def _compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:

        # 1 Create an empty numpy array
        my_strategy_values = np.zeros(self._strategy_space.get_num_dim())

        # 2 Generating random actions based on the dimension of ation space
        ids = self._strategy_space.get_dim_ids()
        for d in range(self._strategy_space.get_num_dim()):
            try:
                base_set = self._strategy_space.get_dim(ids[d]).get_base_set()
            except:
                raise ParamError('Mandatory base set is not defined.')
                
            try:
                if len(self._strategy_space.get_dim(ids[d]).get_boundaries()) == 1:
                    lower_boundaries = 0
                    upper_boundaries = self._strategy_space.get_dim(ids[d]).get_boundaries()[0]
                else:
                    lower_boundaries = self._strategy_space.get_dim(ids[d]).get_boundaries()[0]
                    upper_boundaries = self._strategy_space.get_dim(ids[d]).get_boundaries()[1]
                if base_set == 'Z' or base_set == 'N':
                    my_strategy_values[d] = random.randint(lower_boundaries, upper_boundaries)
                elif base_set == 'R' or base_set == 'DO':
                    my_strategy_values[d] = random.uniform(lower_boundaries, upper_boundaries)
            except:
                raise ParamError('Mandatory boundaries are not defined.')

        # 3 Return an action object with the generated random values
        return GTStrategy(self._id, self._strategy_space, my_strategy_values)