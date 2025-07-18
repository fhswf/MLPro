## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control.controllers
## -- Module  : hunter.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-10  0.1.0     DA       Initial implementation
## -- 2024-12-03  0.1.1     DA       Bugfix
## -- 2025-07-18  0.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2025-07-18)

This module provides a simple demo system that just cumulates a percentage part of the incoming
action to the inner state.
"""


import numpy as np

from mlpro.bf.control import Controller, ControlError, ControlVariable



# Export list for public API
__all__ = [ 'Hunter' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Hunter (Controller):
    """
    ...
    """

    C_NAME              = 'Hunter'

## -------------------------------------------------------------------------------------------------
    def _compute_output(self, p_ctrl_error : ControlError, p_ctrl_var : ControlVariable ):
        p_ctrl_var.values = np.array(p_ctrl_error.values) 
