## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.control.operators
## -- Module  : cumulator.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-06  0.1.0     DA       Creation and initial implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2024-10-06)

This module provides an implementation of a cumulator that determins the next control action by
buffering and cumulating it.

"""

import numpy as np

from mlpro.bf.streams.basics import InstDict
from mlpro.bf.control import Operator



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Cumulator (Operator):
    """
    ...
    """

    C_NAME      = 'Cumulator'

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst: InstDict):
        
        raise NotImplementedError


