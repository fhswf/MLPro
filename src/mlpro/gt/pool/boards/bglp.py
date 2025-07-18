## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.gt.pool.boards
## -- Module  : bglp.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-06-07  0.0.0     SY       Creation
## -- 2021-08-31  1.0.0     SY       Release of first version
## -- 2021-09-01  1.0.1     SY       Minor improvements, code cleaning, add descriptions
## -- 2021-09-06  1.0.2     SY       Minor improvements
## -- 2021-09-11  1.0.3     MRD      Change Header information to match our new library name
## -- 2021-11-29  1.0.4     SY       Enable batch production scenario
## -- 2023-04-12  1.0.5     SY       Refactoring 
## -- 2023-05-11  1.0.6     SY       Refactoring
## -- 2023-06-27  1.0.7     SY       Refactoring module name
## -- 2025-07-17  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-17) 

This module provides an environment of Bulk Good Laboratory Plant (BGLP)
following GT interface. This module provides game board classed based on BGLP
environment of the reinforcement learning pool.
"""

from mlpro.rl.models import Reward
from mlpro.rl.pool.envs.bglp import BGLP
from mlpro.gt import *

        

 # Export list for public API
__all__ = [ 'BGLP_GT' ]


        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class BGLP_GT(BGLP, GameBoard):
    """
    Game theoretical pendant for the reinforcement learning environment class BGLP.
    """

    C_NAME          = 'BGLP_GT'

    def __init__(self, p_logging=True, t_step=0.5, t_set=10.0, demand=0.1,
                 lr_margin=1.0, lr_demand=4.0, lr_power=0.0010, margin_p=[0.2,0.8,4],
                 prod_target=10000, prod_scenario='continuous', cycle_limit=100,
                 p_visualize=False):
        
        BGLP.__init__(self, p_reward_type=Reward.C_TYPE_EVERY_AGENT, p_logging=p_logging,
                      t_step=t_step, t_set=t_set, demand=demand, lr_margin=lr_margin,
                      lr_demand=lr_demand, lr_power=lr_power, margin_p=margin_p,
                      prod_target=prod_target, prod_scenario=prod_scenario,
                      cycle_limit=cycle_limit, p_visualize=p_visualize)


