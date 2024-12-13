## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.plot.backends
## -- Module  : tkagg.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-12-10  0.1.0     DA       Initial implementation
## -- 2024-12-12  0.1.1     DA       Stabilization of method PlotBackendTkAgg.force_foreground()
## -- 2024-12-13  0.2.0     DA       - bugfix in PlotBackendTkAgg._set_geometry()
## --                                - removed methods get_title(), set_title()
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2024-12-13)

This module provides an integration for Matplotlib backend 'TkAgg'.

"""

import re

from mlpro.bf.plot.backends.basics import PlotBackend



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PlotBackendTkAgg (PlotBackend):
    """
    Integrates the Matplotlib backend 'TkAgg' into MLPro.
    """

    C_NAME      = 'TkAgg'

## -------------------------------------------------------------------------------------------------
    def force_foreground(self, p_window):
        p_window.after( 2000, lambda: p_window.attributes('-topmost', True) )


## -------------------------------------------------------------------------------------------------
    def get_pos(self, p_window):
        geometry = p_window.geometry().split('+')
        return int(geometry[1]), int(geometry[2])
    

## -------------------------------------------------------------------------------------------------
    def _set_pos(self, p_window, p_pos : str, p_attempts: int = 10, p_wait: int = 100):
        if p_attempts <= 0: return

        p_window.update_idletasks()            
        p_window.geometry(p_pos)
        p_window.update_idletasks()  

        geo_new = p_window.geometry()

        if p_pos in geo_new: 
            return

        p_window.after( p_wait, lambda: self._set_geometry( p_window = p_window, 
                                                            p_pos = p_pos,
                                                            p_attempts = p_attempts - 1, 
                                                            p_wait = p_wait ) )                


## -------------------------------------------------------------------------------------------------
    def set_pos(self, p_window, p_xpos, p_ypos):
        self._set_pos( p_window = p_window, p_pos = f'+{p_xpos}+{p_ypos}' )
