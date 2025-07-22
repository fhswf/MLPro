## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.pool.native.games
## -- Module  : rockpaperscissors.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-12-08  0.0.0     SY       Creation
## -- 2023-12-08  1.0.0     SY       Release of first version
## -- 2024-01-12  1.0.1     SY       Refactoring: Module Name
## -- 2025-07-18  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-18) 

This module provides a duel of two coalitions of a game of Rock Paper Scissors with random solver.
In the near future, we are going to add more solvers and this howto is going to be updated accordingly.

The game consists of two coaltions, where each coalition makes a decision based on the colllaborative
approach between the coalitions. Each coalition consists of 5 members, the most voted decision of the
5 members represents the final decision of the coalition. 

To be noted, the decision making of the coalitions take place simultaneously, where:
- Decision "0" means Rock
- Decision "1" means Paper
- Decision "2" means Scissors

"""

import numpy as np

from mlpro.bf.math import Dimension, MSpace
from mlpro.bf.ml import Model 

from mlpro.gt.native.basics import *
from mlpro.gt.pool.native.solvers.randomsolver import RandomSolver
         
        
        
# Export list for public API
__all__ = [ 'PayoffFunction_RSP',
            'RockPaperScissors' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PayoffFunction_RSP (GTFunction):

## -------------------------------------------------------------------------------------------------
    def _setup_mapping_matrix(self) -> np.ndarray:

        mapping = np.array([[[0,0], [0,1], [0,2]], [[1,0], [1,1], [1,2]], [[2,0], [2,1], [2,2]]])
        
        return mapping


## -------------------------------------------------------------------------------------------------
    def _setup_payoff_matrix(self):

        self._add_payoff_matrix(
            p_idx=0,
            p_payoff_matrix=np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        )

        self._add_payoff_matrix(
            p_idx=1,
            p_payoff_matrix=np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        )
         
        
        


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RockPaperScissors (GTGame):

    C_NAME  = 'RockPaperScissors'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:
        
        _strategy_space = MSpace()
        _strategy_space.add_dim(Dimension('RStr','Z','Random Strategy','','','',[0,2]))
        
        solver1 = RandomSolver(
            p_strategy_space=_strategy_space,
            p_id=1,
            p_name="Random Solver",
            p_visualize=p_visualize,
            p_logging=p_logging
        )


        p1_1 = GTPlayer(
            p_solver=solver1,
            p_name="Player 1 of Team 1",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=False
        )


        p1_2 = GTPlayer(
            p_solver=solver1,
            p_name="Player 2 of Team 1",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=False
        )


        p1_3 = GTPlayer(
            p_solver=solver1,
            p_name="Player 3 of Team 1",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=False
        )


        p1_4 = GTPlayer(
            p_solver=solver1,
            p_name="Player 4 of Team 1",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=False
        )


        p1_5 = GTPlayer(
            p_solver=solver1,
            p_name="Player 5 of Team 1",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=False
        )


        coal1 = GTCoalition(
            p_name="Coalition of Team 1",
            p_coalition_type=GTCoalition.C_COALITION_MODE
        )
        coal1.add_player(p1_1)
        coal1.add_player(p1_2)
        coal1.add_player(p1_3)
        coal1.add_player(p1_4)
        coal1.add_player(p1_5)


        solver2 = RandomSolver(
            p_strategy_space=_strategy_space,
            p_id=2,
            p_visualize=p_visualize,
            p_logging=p_logging
        )

        p2_1 = GTPlayer(
            p_solver=solver2,
            p_name="Player 1 of Team 2",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=False
        )

        p2_2 = GTPlayer(
            p_solver=solver2,
            p_name="Player 2 of Team 2",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=False
        )

        p2_3 = GTPlayer(
            p_solver=solver2,
            p_name="Player 3 of Team 2",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=False
        )

        p2_4 = GTPlayer(
            p_solver=solver2,
            p_name="Player 4 of Team 2",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=False
        )

        p2_5 = GTPlayer(
            p_solver=solver2,
            p_name="Player 5 of Team 2",
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_random_solver=False
        )

        coal2 = GTCoalition(
            p_name="Coalition of Team 2",
            p_coalition_type=GTCoalition.C_COALITION_MODE
        )
        coal2.add_player(p2_1)
        coal2.add_player(p2_2)
        coal2.add_player(p2_3)
        coal2.add_player(p2_4)
        coal2.add_player(p2_5)


        competition = GTCompetition(
            p_name="Rock Paper Scissors Competition",
            p_logging=p_logging
            )
        competition.add_coalition(coal1)
        competition.add_coalition(coal2)
        
        coal_ids = competition.get_coalitions_ids()

        self._payoff = GTPayoffMatrix(
            p_function=PayoffFunction_RSP(
                p_func_type=GTFunction.C_FUNC_PAYOFF_MATRIX,
                p_dim_elems=[3,3],
                p_num_coalisions=2
                ),
            p_player_ids=coal_ids
        )
        
        return competition