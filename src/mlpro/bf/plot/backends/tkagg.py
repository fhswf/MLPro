## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.plot.backends
## -- Module  : tkagg.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-12-10  0.1.0     DA       Initial implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2024-12-10)

This module provides various classes related to data plotting.

"""


from mlpro.bf.plot.backends.basics import PlotBackend



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PlotBackendTkAgg (PlotBackend):

    C_NAME      = 'TkAgg'

## -------------------------------------------------------------------------------------------------
    def force_foreground(self, p_window):
        p_window.attributes('-topmost', True) 


## -------------------------------------------------------------------------------------------------
    def _set_geometry(self, p_window, p_geo, p_attempts: int = 10, p_wait: int = 100):
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
    def set_title(self, p_window, p_title):
        p_window.title(p_title)
