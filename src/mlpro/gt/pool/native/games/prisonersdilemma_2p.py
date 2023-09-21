## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.pool.native.games
## -- Module  : prisonersdilemma_2p
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-09-21  0.0.0     SY       Creation
## -- 2023-xx-xx  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-xx-xx)

This module provides a 2-player game of Prisoners' Dilemma with random solver. In the near future,
we are going to add more solvers and this howto is going to be updated accordingly.
"""

from mlpro.gt.native.basics import *
from mlpro.gt.pool.native.solvers.randomsolver import RandomSolver
         
        
        


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PayoffFunction_PD2P (GTFunction):


## -------------------------------------------------------------------------------------------------
    def _setup_payoff_matrix(self):

        raise NotImplementedError
         
        
        



