## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.gt.examples
## -- Module  : howto_gt_native_005_routing_problems_3p.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-01-12  0.0.0     SY       Creation
## -- 2024-01-12  1.0.0     SY       Release of first version
## -- 2025-07-18  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-18) 

This module shows how to run a 3P competition game of routing problems.

You will learn:
    
1) How to set up a game, including solver, competition, coalition, payoff, and more
    
2) How to run the game

3) How to analyse the game
    
"""

from pathlib import Path

from mlpro.bf import Log

from mlpro.gt.native import *
from mlpro.gt.pool.native.games.routingproblems_3p import *




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
        p_game_cls=Routing_3P,
        p_cycle_limit=cycle_limit,
        p_path=path,
        p_visualize=visualize,
        p_logging=logging
        )

training.run()