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
    def _figure_force_foreground_default(self, p_figure):
        window = p_figure.canvas.manager.window
        window.after( 2000, lambda: window.attributes('-topmost', True) )


## -------------------------------------------------------------------------------------------------
    def _figure_get_geometry_Windows(self, p_figure):
        geometry = p_figure.canvas.manager.window.geometry().split('+')
        xpos = int(geometry[1])
        ypos = int(geometry[2])

        size = p_figure.get_size_inches()
        width = size[0]
        height = size[1]

        return xpos, ypos, width, height
    

## -------------------------------------------------------------------------------------------------
    def _figure_get_geometry_default(self, p_figure):
        geometry = p_figure.canvas.manager.window.geometry()
        numbers = list(map(int, re.findall(r'\d+', geometry)))
        return numbers[2], numbers[3], numbers[0], numbers[1]
    

    ## -------------------------------------------------------------------------------------------------
    def _figure_set_pos_Windows_rec(self, p_window, p_pos: str, p_attempts: int = 10, p_wait: int = 100):
        if p_attempts <= 0: return

        p_window.update_idletasks()            
        p_window.geometry(p_pos)
        p_window.update_idletasks()  

        match = re.search(r'\+\d+\+\d+', p_window.geometry())
        if match:
            pos_new = match.group()

        if p_pos == pos_new: 
            return

        p_window.after( p_wait, lambda: self._figure_set_pos_Windows_rec( p_window = p_window, 
                                                                          p_pos = p_pos,
                                                                          p_attempts = p_attempts - 1, 
                                                                          p_wait = p_wait ) )            


## -------------------------------------------------------------------------------------------------
    def _figure_set_geometry_Windows(self, p_figure, p_xpos, p_ypos, p_width, p_height):
        p_figure.set_size_inches( w = p_width, h = p_height)
        self._figure_set_pos_Windows_rec( p_window=p_figure.canvas.manager.window, p_pos=f'+{p_xpos}+{p_ypos}')


## -------------------------------------------------------------------------------------------------
    def _figure_set_geometry_default_rec(self, p_window, p_geometry: str, p_attempts: int = 10, p_wait: int = 100):
        if p_attempts <= 0: return

        p_window.update_idletasks()            
        p_window.geometry(p_geometry)
        p_window.update_idletasks()  

        geo_new = p_window.geometry()

        if p_geometry == geo_new: 
            return

        p_window.after( p_wait, lambda: self._figure_set_geometry_default_rec( p_window = p_window, 
                                                                               p_geometry = p_geometry,
                                                                               p_attempts = p_attempts - 1, 
                                                                               p_wait = p_wait ) )             


## -------------------------------------------------------------------------------------------------
    def _figure_set_geometry_default(self, p_figure, p_xpos, p_ypos, p_width, p_height):
        geometry = f'{p_width}x{p_height}+{p_xpos}+{p_ypos}'
        self._figure_set_geometry_default_rec( p_window = p_figure.canvas.manager.window, p_geometry=geometry )