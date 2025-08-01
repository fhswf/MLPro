## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.gt.examples
## -- Module  : howto_gt_native_002_prisoners_dilemma_3p.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-12-07  0.0.0     SY       Creation
## -- 2023-12-12  1.0.0     SY       Release of first version
## -- 2024-01-05  1.0.1     SY       Renaming
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2024-01-05)

This module shows how to run a game, namely 3P Prisoners' Dilemma with two solvers, such as random
solver and min greedy policy.

You will learn:
    
1) How to set up a game, including solver, competition, coalition, payoff, and more
    
2) How to run the game

3) How to analyse the game
    
"""

from pathlib import Path

from mlpro.bf import Log
from mlpro.gt.native import *
from mlpro.gt.pool.native.games.prisonersdilemma_3p import *




if __name__ == "__main__":
    cycle_limit = 10
    logging     = Log.C_LOG_ALL
    visualize   = False
    path        = str(Path.home())

else:
    cycle_limit = 1
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    path        = None

training = GTTraining(
        p_game_cls=PrisonersDilemma3PGame,
        p_cycle_limit=cycle_limit,
        p_path=path,
        p_visualize=visualize,
        p_logging=logging
        )

training.run()