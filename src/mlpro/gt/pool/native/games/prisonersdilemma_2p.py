## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.pool.native.games
## -- Module  : prisonersdilemma_2p.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-09-21  0.0.0     SY       Creation
## -- 2023-12-08  1.0.0     SY       Release of first version
## -- 2024-01-12  1.0.1     SY       Refactoring: Module Name
## -- 2024-01-27  1.0.2     SY       Refactoring: Payoff Matrix
## -- 2025-07-18  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-18) 

This module provides a 2-player game of Prisoners' Dilemma with random solver. In the near future,
we are going to add more solvers and this howto is going to be updated accordingly.

The game consists of two competitors, where each competitor represents a prisonner.
Both of them have a goal to minimize their prison sentences, where their length of sentences depend
on their decision in front of the jury.

If a prisoner pleads guilty, while another prisoner pleads not guilty. The guilty prisoner gets 8 years
of imprisonment, while the not guilty prisoner gets 1 year of imprisonment.

If both of them plead guilty, then each of them gets 2 years of imprisonment.

Meanwhile, if both of them plead not guilty, then each of them obtains 5 years of imprisonment.

To be noted, the decision making of the prisoners take place simultaneously, where:
- Decision "0" means confess
- Decision "1" means not confess

"""

import numpy as np

from mlpro.bf.math import Dimension, MSpace
from mlpro.bf.ml import Model  

from mlpro.gt.native.basics import *
from mlpro.gt.pool.native.solvers.randomsolver import RandomSolver
         
        
        
# Export list for public API
__all__ = [ 'PayoffFunction_PD2P', 
            'PrisonersDilemma2PGame' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PayoffFunction_PD2P (GTFunction):

## -------------------------------------------------------------------------------------------------
    def _setup_mapping_matrix(self) -> np.ndarray:

        mapping = np.array([[[0,0], [0,1]], [[1,0], [1,1]]])
        
        return mapping


## -------------------------------------------------------------------------------------------------
    def _setup_payoff_matrix(self):

        self._add_payoff_matrix(
            p_idx=0,
            p_payoff_matrix=np.array([[2, 8], [1, 5]]) # ([[(0,0), (0,1)], [(1,0), (1,1)]])
        )

        self._add_payoff_matrix(
            p_idx=1,
            p_payoff_matrix=np.array([[2, 1], [8, 5]])
        )
         
        
        


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PrisonersDilemma2PGame (GTGame):

    C_NAME  = 'PrisonersDilemma2PGame'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:
        
        _strategy_space = MSpace()
        _strategy_space.add_dim(Dimension('RStr','Z','Random Strategy','','','',[0,1]))
        
        solver1 = RandomSolver(
            p_strategy_space=_strategy_space,
            p_id=1,
            p_name="Random Solver",
            p_visualize=p_visualize,
            p_logging=p_logging
        )


        p1 = GTPlayer(
            p_solver=solver1,
            p_name="Player of Prisoner 1",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=False
        )

        coal1 = GTCoalition(
            p_name="Coalition of Prisoner 1",
            p_coalition_type=GTCoalition.C_COALITION_SUM
        )
        coal1.add_player(p1)


        solver2 = RandomSolver(
            p_strategy_space=_strategy_space,
            p_id=2,
            p_visualize=p_visualize,
            p_logging=p_logging
        )

        p2 = GTPlayer(
            p_solver=solver2,
            p_name="Player of Prisoner 2",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=False
        )

        coal2 = GTCoalition(
            p_name="Coalition of Prisoner 2",
            p_coalition_type=GTCoalition.C_COALITION_SUM
        )
        coal2.add_player(p2)


        competition = GTCompetition(
            p_name="Prisoner's Dilemma Competition",
            p_logging=p_logging
            )
        competition.add_coalition(coal1)
        competition.add_coalition(coal2)
        
        coal_ids = competition.get_coalitions_ids()

        self._payoff = GTPayoffMatrix(
            p_function=PayoffFunction_PD2P(
                p_func_type=GTFunction.C_FUNC_PAYOFF_MATRIX,
                p_dim_elems=[2,2],
                p_num_coalisions=2
                ),
            p_player_ids=coal_ids
        )
        
        return competition
        



