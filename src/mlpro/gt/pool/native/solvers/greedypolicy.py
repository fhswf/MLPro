## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.pool.native.solvers
## -- Module  : greedypolicy.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-12-07  0.0.0     SY       Creation
## -- 2023-12-12  1.0.0     SY       Release of first version
## -- 2024-01-18  1.0.1     SY       Refactoring: Module Name, MinGreedyPolicy
## -- 2025-07-18  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-18) 

This module provides solver with greedy GT strategy. There are two variants, such as minimum greedy
and maximum greedy.
"""

import numpy as np

from mlpro.gt.native.basics import *
         
        

# Export list for public API
__all__ = [ 'MaxGreedyPolicy',
            'MinGreedyPolicy' ]

 


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MaxGreedyPolicy (GTSolver):
    """
    A solver that generates actions for each dimension of the underlying strategy space based on the
    maximum greedy policy.
    """

    C_NAME      = 'MaxGreedyPolicy'


## -------------------------------------------------------------------------------------------------
    def _compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:

        if p_payoff._function.C_FUNCTION_TYPE == p_payoff._function.C_FUNC_PAYOFF_MATRIX:
            stg_values              = np.zeros(self._strategy_space.get_num_dim())

            idx                     = self.get_id()-1
            payoff_matrix           = p_payoff._function._payoff_map[idx]
            best_payoff             = np.max(p_payoff._function._payoff_map[idx])
            
            for p,x in enumerate(payoff_matrix):
                try:
                    y               = payoff_matrix[p].tolist().index(best_payoff)
                    stg             = p_payoff._function._mapping_matrix[p][y]
                    stg_values[0]   = stg[idx]
                    return GTStrategy(self._id, self._strategy_space, stg_values)
                except:
                    pass
        else:
            return self._call_compute_strategy(p_payoff)


## -------------------------------------------------------------------------------------------------
    def _call_compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:
        
        raise NotImplementedError
         
        
        


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MinGreedyPolicy (GTSolver):
    """
    A solver that generates actions for each dimension of the underlying strategy space based on the
    minimum greedy policy.
    """

    C_NAME      = 'MinGreedyPolicy'


## -------------------------------------------------------------------------------------------------
    def _compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:

        if p_payoff._function.C_FUNCTION_TYPE == p_payoff._function.C_FUNC_PAYOFF_MATRIX:
            stg_values              = np.zeros(self._strategy_space.get_num_dim())

            idx                     = self.get_id()-1
            payoff_matrix           = p_payoff._function._payoff_map[idx]
            least_payoff            = np.min(p_payoff._function._payoff_map[idx])
            
            for p,x in enumerate(payoff_matrix):
                try:
                    y               = payoff_matrix[p].tolist().index(least_payoff)
                    stg             = p_payoff._function._mapping_matrix[p][y]
                    stg_values[0]   = stg[idx]
                    return GTStrategy(self._id, self._strategy_space, stg_values)
                except:
                    pass
        else:
            return self._call_compute_strategy(p_payoff)


## -------------------------------------------------------------------------------------------------
    def _call_compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:
        
        raise NotImplementedError