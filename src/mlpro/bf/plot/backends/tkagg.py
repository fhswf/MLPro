## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.plot.backends
## -- Module  : tkagg.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-12-10  0.1.0     DA       Initial implementation
## -- 2024-12-12  0.1.1     DA       Stabilization of method PlotBackendTkAgg.force_foreground()
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.1 (2024-12-12)

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
    def get_geometry(self, p_window):
        geometry = p_window.geometry()
        pattern  = r"(\d+)x(\d+)\+(\d+)\+(\d+)"

        match = re.match(pattern, geometry)

        if match:
            return map(int, match.groups())
        else:
            return None


## -------------------------------------------------------------------------------------------------
    def _set_geometry(self, p_window, p_geo : str, p_attempts: int = 10, p_wait: int = 100):
        if p_attempts <= 0: return

        p_window.update_idletasks()            
        p_window.geometry(p_geo)
        p_window.update_idletasks()  

        geo_new = p_window.geometry()

        if geo_new == p_geo: return

        p_window.after( p_wait, lambda: self._set_geometry( p_window = p_window, 
                                                            p_attempts = p_attempts - 1, 
                                                            p_wait = p_wait ) )                


## -------------------------------------------------------------------------------------------------
    def set_geometry(self, p_window, p_xpos, p_ypos, p_width, p_height):

        geo = str(p_width) + 'x' + str(p_height) + '+' + str(p_xpos) + '+' + str(p_ypos)

        self._set_geometry( p_window = p_window, p_geo = geo )


## -------------------------------------------------------------------------------------------------
    def get_title(self, p_window):
        return p_window.title()


## -------------------------------------------------------------------------------------------------
    def set_title(self, p_window, p_title: str):
        p_window.title(p_title)
