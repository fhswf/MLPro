## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.plot.backends
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-12-10  0.1.0     DA       Initial implementation
## -- 2924-12-13  0.2.0     DA       Removed methods get_title(), set_title()
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2024-12-13)

This module provides the template class PlotBackend for special support of common Matplotlib backends.

"""



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PlotBackend:
    
    C_NAME      = '????'

## -------------------------------------------------------------------------------------------------
    def force_foreground(self, p_window):
        pass


## -------------------------------------------------------------------------------------------------
    def get_pos(self, p_window):
        pass


## -------------------------------------------------------------------------------------------------
    def set_pos(self, p_window, p_xpos, p_ypos):
        pass
