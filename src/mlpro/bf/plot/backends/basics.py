## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.plot.backends
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-12-10  0.1.0     DA       Initial implementation
## -- 2024-12-13  0.2.0     DA       Removed methods get_title(), set_title()
## -- 2024-12-16  0.3.0     DA       Introduction of platform-dependant methods
## -- 2024-12-30  0.4.0     DA       Class PlotBackend: new set of methods atexit*()
## -- 2025-01-03  0.5.0     DA       New class WindowGeometry
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.5.0 (2025-01-03)

This module provides the template class PlotBackend for special support of common Matplotlib backends.

"""


import platform
import atexit
from typing import TypedDict, Union

try:
    from matplotlib.figure import Figure
except:
    class Figure : pass


# Export list for public API
__all__ = [ 'WindowState',
            'WSMINIMIZED',
            'WSNORMAL',
            'WSMAXIMIZED',
            'WindowGeometry',
            'PlotBackend' ]



WindowState = int
WSMINIMIZED = 0
WSNORMAL    = 1
WSMAXIMIZED = 2




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WindowGeometry (TypedDict, total=True):
    xpos   : Union[int, float]
    ypos   : Union[int, float]
    width  : Union[int, float]
    height : Union[int, float]
    state  : WindowState





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PlotBackend:
    
    C_NAME      = '????'

## -------------------------------------------------------------------------------------------------
    def __init__(self):
        self._os = platform.system()  # 'Linux', 'Windows', 'Darwin'

        self.figure_force_foreground = getattr(self, '_figure_force_foreground_' + self._os, self._figure_force_foreground_default )
        self.figure_get_title        = getattr(self, '_figure_get_title_' + self._os, self._figure_get_title_default )
        self.figure_set_title        = getattr(self, '_figure_set_title_' + self._os, self._figure_set_title_default )
        self.figure_get_geometry     = getattr(self, '_figure_get_geometry_' + self._os, self._figure_get_geometry_default )
        self.figure_set_geometry     = getattr(self, '_figure_set_geometry_' + self._os, self._figure_set_geometry_default )
        self.figure_atexit           = getattr(self, '_figure_atexit_' + self._os, self._figure_atexit_default )


## -------------------------------------------------------------------------------------------------
    def _figure_force_foreground_Linux(self, p_figure : Figure):
        self._figure_force_foreground_default( p_figure=p_figure)


## -------------------------------------------------------------------------------------------------
    def _figure_force_foreground_Windows(self, p_figure : Figure):
        self._figure_force_foreground_default( p_figure=p_figure)


## -------------------------------------------------------------------------------------------------
    def _figure_force_foreground_Darwin(self, p_figure : Figure):
        self._figure_force_foreground_default( p_figure=p_figure)


## -------------------------------------------------------------------------------------------------
    def _figure_force_foreground_default(self, p_figure : Figure):
        raise NotImplementedError
    

## -------------------------------------------------------------------------------------------------
    def figure_force_foreground(self, p_figure : Figure):
        pass


## -------------------------------------------------------------------------------------------------
    def _figure_get_title_Linux(self, p_figure : Figure) -> str:
        return self._figure_get_title_default(p_figure = p_figure)


## -------------------------------------------------------------------------------------------------
    def _figure_get_title_Windows(self, p_figure : Figure) -> str:
        return self._figure_get_title_default(p_figure = p_figure)


## -------------------------------------------------------------------------------------------------
    def _figure_get_title_Darwin(self, p_figure : Figure) -> str:
        return self._figure_get_title_default(p_figure = p_figure)


## -------------------------------------------------------------------------------------------------
    def _figure_get_title_default(self, p_figure : Figure) -> str:
        return p_figure.canvas.manager.get_window_title()


## -------------------------------------------------------------------------------------------------
    def figure_get_title(self, p_figure : Figure) -> str:
        pass
    
    
## -------------------------------------------------------------------------------------------------
    def _figure_set_title_Linux(self, p_figure : Figure, p_title : str):
        self._figure_set_title_default( p_figure = p_figure, p_title = p_title )


## -------------------------------------------------------------------------------------------------
    def _figure_set_title_Windows(self, p_figure : Figure, p_title : str):
        self._figure_set_title_default( p_figure = p_figure, p_title = p_title )


## -------------------------------------------------------------------------------------------------
    def _figure_set_title_Darwin(self, p_figure : Figure, p_title : str):
        self._figure_set_title_default( p_figure = p_figure, p_title = p_title )


## -------------------------------------------------------------------------------------------------
    def _figure_set_title_default(self, p_figure : Figure, p_title : str):
        p_figure.canvas.manager.set_window_title( p_title )


## -------------------------------------------------------------------------------------------------
    def figure_set_title(self, p_figure : Figure, p_title : str):
        pass


## -------------------------------------------------------------------------------------------------
    def _figure_get_geometry_Linux(self, p_figure : Figure) -> WindowGeometry:
        return self._figure_get_geometry_default( p_figure = p_figure )


## -------------------------------------------------------------------------------------------------
    def _figure_get_geometry_Windows(self, p_figure : Figure) -> WindowGeometry:
        return self._figure_get_geometry_default( p_figure = p_figure )


## -------------------------------------------------------------------------------------------------
    def _figure_get_geometry_Darwin(self, p_figure : Figure) -> WindowGeometry:
        return self._figure_get_geometry_default( p_figure = p_figure )


## -------------------------------------------------------------------------------------------------
    def _figure_get_geometry_default(self, p_figure : Figure) -> WindowGeometry:
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def figure_get_geometry(self, p_figure : Figure) -> WindowGeometry:
        pass


## -------------------------------------------------------------------------------------------------
    def _figure_set_geometry_Linux( self, p_figure : Figure, p_geometry : WindowGeometry ):
        self._figure_set_geometry_default( p_figure = p_figure, p_geometry = p_geometry )


## -------------------------------------------------------------------------------------------------
    def _figure_set_geometry_Windows( self, p_figure : Figure, p_geometry : WindowGeometry  ):
        self._figure_set_geometry_default( p_figure = p_figure, p_geometry = p_geometry )


## -------------------------------------------------------------------------------------------------
    def _figure_set_geometry_Darwin( self, p_figure : Figure, p_geometry : WindowGeometry  ):
        self._figure_set_geometry_default( p_figure = p_figure, p_geometry = p_geometry )


## -------------------------------------------------------------------------------------------------
    def _figure_set_geometry_default( self, p_figure : Figure, p_geometry : WindowGeometry  ):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def figure_set_geometry( self, p_figure : Figure, p_geometry : WindowGeometry ):
        pass

## -------------------------------------------------------------------------------------------------
    def _figure_atexit_Linux(self, p_figure : Figure, p_fct):
        self._figure_atexit_default( p_figure = p_figure, p_fct=p_fct)


## -------------------------------------------------------------------------------------------------
    def _figure_atexit_Windows(self, p_figure : Figure, p_fct):
        self._figure_atexit_default( p_figure = p_figure, p_fct=p_fct)


## -------------------------------------------------------------------------------------------------
    def _figure_atexit_Darwin(self, p_figure : Figure, p_fct):
        self._figure_atexit_default( p_figure = p_figure, p_fct=p_fct)


## -------------------------------------------------------------------------------------------------
    def _figure_atexit_default(self, p_figure : Figure, p_fct):
        atexit.register(p_fct)


## -------------------------------------------------------------------------------------------------
    def figure_atexit(self, p_figure : Figure, p_fct):
        pass