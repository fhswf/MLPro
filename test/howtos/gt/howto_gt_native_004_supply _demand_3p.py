## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.gt.examples
## -- Module  : howto_gt_native_004_supply _demand_3p.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-12-12  0.0.0     SY       Creation
## -- 2023-12-12  1.0.0     SY       Release of first version
## -- 2024-01-12  1.0.1     SY       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2024-01-12)

This module shows how to run a 3 sellers competition game of supply and demand.

You will learn:
    
1) How to set up a game, including solver, competition, coalition, payoff, and more
    
2) How to run the game

3) How to analyse the game
    
"""

from mlpro.gt.native.basics import *
from mlpro.gt.pool.native.games.supplydemand_3p import *
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
        p_game_cls=SupplyDemand_3P,
        p_cycle_limit=cycle_limit,
        p_path=path,
        p_visualize=visualize,
        p_logging=logging
        )

training.run()