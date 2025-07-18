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
## -- 2024-12-16  0.3.0     DA       Refactoring and validation for Linux
## -- 2025-01-03  0.4.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.4.0 (2025-01-03)

This module provides an integration for Matplotlib backend 'TkAgg'.

"""


import re

try:
    from matplotlib.figure import Figure
except:
    class Figure : pass

from mlpro.bf.plot.backends import PlotBackend, WindowGeometry, WindowState, WSNORMAL, WSMINIMIZED, WSMAXIMIZED



# Export list for public API
__all__ = [ 'PlotBackendTkAgg' ]




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
        # p_figure.canvas.manager.window.attributes('-topmost', True) 


## -------------------------------------------------------------------------------------------------
    def _figure_get_geometry_Windows(self, p_figure) -> WindowGeometry:

        # 1 Get size and position   
        window   = p_figure.canvas.manager.window     
        geometry = window.geometry().split('+')
        xpos = int(geometry[1])
        ypos = int(geometry[2])

        size   = p_figure.get_size_inches()
        width  = size[0]
        height = size[1]

        # 2 Get window state
        state_tk = window.wm_state()
        if state_tk == 'iconic':
            state = WSMINIMIZED
        elif state_tk == 'zoomed': 
            state = WSMAXIMIZED
        else:
            state = WSNORMAL

        # 3 Return dictionary compatible with type WindowGeometry
        return { 'xpos'   : xpos,
                 'ypos'   : ypos,
                 'width'  : width,
                 'height' : height,
                 'state'  : state }
    

## -------------------------------------------------------------------------------------------------
    def _figure_get_geometry_default(self, p_figure) -> WindowGeometry:

        # 1 Get size and position  
        window   = p_figure.canvas.manager.window
        geometry = window.geometry()
        numbers  = list(map(int, re.findall(r'\d+', geometry)))

        # 2 Get window state
        state_tk = window.wm_state()
        if state_tk == 'iconic':
            state = WSMINIMIZED
        elif state_tk == 'zoomed': 
            state = WSMAXIMIZED
        else:
            state = WSNORMAL

        # 3 Return dictionary compatible with type WindowGeometry
        return { 'xpos'   : numbers[2],
                 'ypos'   : numbers[3],
                 'width'  : numbers[0],
                 'height' : numbers[1],
                 'state'  : state }
    

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
    def _figure_set_geometry_Windows(self, p_figure : Figure, p_geometry : WindowGeometry):

        # 1 Set size and position
        window=p_figure.canvas.manager.window
        p_figure.set_size_inches( w = p_geometry['width'], h = p_geometry['height'])
        self._figure_set_pos_Windows_rec( p_window=window, p_pos=f'+{p_geometry["xpos"]}+{p_geometry["ypos"]}')

        # 2 Set window state
        state = p_geometry['state']
        if state == WSMINIMIZED:
            state_tk = 'iconic'
        elif state == WSMAXIMIZED:
            state_tk = 'zoomed'
        else:
            state_tk = 'normal'

        window.wm_state(state_tk)


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
    def _figure_set_geometry_default(self, p_figure : Figure, p_geometry : WindowGeometry):

        # 1 Set size and position        
        window=p_figure.canvas.manager.window
        geometry_tk = f'{p_geometry["width"]}x{p_geometry["height"]}+{p_geometry["xpos"]}+{p_geometry["ypos"]}'
        self._figure_set_geometry_default_rec( p_window = window, p_geometry=geometry_tk )

        # 2 Set window state
        state = p_geometry['state']
        if state == WSMINIMIZED:
            state_tk = 'iconic'
        elif state == WSMAXIMIZED:
            state_tk = 'zoomed'
        else:
            state_tk = 'normal'

        window.wm_state(state_tk)