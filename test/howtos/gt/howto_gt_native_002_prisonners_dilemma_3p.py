## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.gt.examples
## -- Module  : howto_gt_native_002_prisonners_dilemma_3p.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-12-07  0.0.0     SY       Creation
## -- 2023-12-12  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-12-12)

This module shows how to run a game, namely 3P Prisoners' Dilemma with two solvers, such as random
solver and min greedy policy.

You will learn:
    
1) How to set up a game, including solver, competition, coalition, payoff, and more
    
2) How to run the game

3) How to analyse the game
    
"""

from mlpro.gt.native.basics import *
from mlpro.gt.pool.native.games.prisonersdilemma_3p import *
from pathlib import Path



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