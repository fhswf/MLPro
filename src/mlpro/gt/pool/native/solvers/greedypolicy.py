## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.pool.native.solvers
## -- Module  : greedypolicy
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-12-07  0.0.0     SY       Creation
## -- 2023-12-07  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-12-07)

This module provides solver with greedy GT strategy. There are two variants, such as minimum greedy
and maximum greedy.
"""

from mlpro.gt.native.basics import *
         
        
        


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

        if p_payoff._function is not None:
            my_strategy_values = np.zeros(self._strategy_space.get_num_dim())

            idx = self.get_id()-1
            id = p_payoff._player_ids[self.get_id()-1]
            best_payoff = p_payoff._function.best_response(id)
            payoff_matrix = p_payoff._function._payoff_map[idx]
            my_strategy_values[0] = np.where(payoff_matrix==best_payoff)[idx].item()
            return GTStrategy(self._id, self._strategy_space, my_strategy_values)
        else:
            return self._call_compute_strategy()


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

        if p_payoff._function is not None:
            my_strategy_values = np.zeros(self._strategy_space.get_num_dim())

            idx = self.get_id()-1
            id = p_payoff._player_ids[self.get_id()-1]
            payoff_matrix = p_payoff._function._payoff_map[idx]
            least_payoff = np.min(self._payoff_map[id])
            my_strategy_values[0] = np.where(payoff_matrix==least_payoff)[idx].item()
            return GTStrategy(self._id, self._strategy_space, my_strategy_values)
        else:
            return self._call_compute_strategy()


## -------------------------------------------------------------------------------------------------
    def _call_compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:
        
        raise NotImplementedError