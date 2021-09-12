## -----------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : BGLP_GT
## -----------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.  Auth.  Description
## -- 2021-06-07  0.00   SY     Creation
## -- 2021-08-31  1.00   SY     Release of first version
## -- 2021-09-01  1.01   SY     Minor improvements, code cleaning, add descriptions
## -- 2021-09-06  1.02   SY     Minor improvements
## -- 2021-09-11  1.02  MRD    Change Header information to match our new library name
## -----------------------------------------------------------------------------

"""
Ver. 1.02 (2021-09-06)

This module provides an environment of Bulk Good Laboratory Plant (BGLP)
following GT interface. This module provides game board classed based on BGLP
environment of the reinforcement learning pool.
"""

from mlpro.rl.models import Reward
from mlpro.rl.pool.envs.bglp import BGLP
from mlpro.gt.models import *

        
        
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class BGLP_GT(BGLP, GameBoard):
    """
    Game theoretical pendant for the reinforcement learning environment class BGLP.
    """

    C_NAME          = 'BGLP_GT'

    def __init__(self, p_logging=True,t_step=0.5, t_set=10.0, demand=0.1,
                 lr_margin=1.0, lr_demand=4.0, lr_energy=0.0010, margin_p=[0.2,0.8,4]):
        BGLP.__init__(self, p_reward_type=Reward.C_TYPE_EVERY_AGENT, p_logging=p_logging,
                      t_step=t_step, t_set=t_set, demand=demand, lr_margin=lr_margin,
                      lr_demand=lr_demand, lr_energy=lr_energy, margin_p=margin_p)


