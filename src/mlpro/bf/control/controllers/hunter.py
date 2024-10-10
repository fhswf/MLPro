## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control.controllers
## -- Module  : hunter.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-10  0.1.0     DA       Initial implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2024-10-10)

This module provides a simple demo system that just cumulates a percentage part of the incoming
action to the inner state.
"""


import numpy as np

from mlpro.bf.systems import ActionElement
from mlpro.bf.systems.pool import Fox
from mlpro.bf.control import Controller, ControlError



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Hunter (Controller):
    """
    ...
    """

    C_NAME              = 'Hunter'

## -------------------------------------------------------------------------------------------------
    def _compute_output(self, p_ctrl_error : ControlError, p_ctrl_var_elem : ActionElement):
        p_ctrl_var_elem.values = np.array(p_ctrl_error.values) * (-1)