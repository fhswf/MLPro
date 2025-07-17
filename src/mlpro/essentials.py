## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro
## -- Module  : essentials
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-07-17  1.0.0     DA       Creation/release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-07-17)

This module provides simplified access to essential MLPro classes.
"""


from mlpro.bf.various import Log
from mlpro.bf.ops import Mode
from mlpro.bf.plot import PlotSettings


# Export list for public API
__all__ = [ 'Log',
            'Mode',
             'PlotSettings' ]
