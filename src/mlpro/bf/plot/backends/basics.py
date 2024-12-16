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
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.3.0 (2024-12-16)

This module provides the template class PlotBackend for special support of common Matplotlib backends.

"""


import platform

try:
    from matplotlib.figure import Figure
except:
    class Figure : pass



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PlotBackend:
    
    C_NAME      = '????'

## -------------------------------------------------------------------------------------------------
    def __init__(self):
        self._os = platform.system()  # 'Linux', 'Windows', 'Darwin'

        self.figure_force_foreground = getattr(self, '_figure_force_foreground_' + self._os, self._figure_force_foreground_default )
        self.figure_get_title = getattr(self, '_figure_get_title_' + self._os, self._figure_get_title_default )
        self.figure_set_title = getattr(self, '_figure_set_title_' + self._os, self._figure_set_title_default )
        self.figure_get_geometry = getattr(self, '_figure_get_geometry_' + self._os, self._figure_get_geometry_default )
        self.figure_set_geometry = getattr(self, '_figure_set_geometry_' + self._os, self._figure_set_geometry_default )


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
        raise NotImplementedError
    
    
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
    def _figure_get_geometry_Linux(self, p_figure : Figure):
        return self._figure_get_geometry_default( p_figure = p_figure )


## -------------------------------------------------------------------------------------------------
    def _figure_get_geometry_Windows(self, p_figure : Figure):
        return self._figure_get_geometry_default( p_figure = p_figure )


## -------------------------------------------------------------------------------------------------
    def _figure_get_geometry_Darwin(self, p_figure : Figure):
        return self._figure_get_geometry_default( p_figure = p_figure )


## -------------------------------------------------------------------------------------------------
    def _figure_get_geometry_default(self, p_figure : Figure):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def figure_get_geometry(self, p_figure : Figure):
        pass


## -------------------------------------------------------------------------------------------------
    def _figure_set_geometry_Linux( self, p_figure : Figure, p_xpos, p_ypos, p_width, p_height ):
        self._figure_set_geometry_default( p_figure = p_figure, p_xpos = p_xpos, p_ypos = p_ypos, p_width = p_width, p_height = p_height)


## -------------------------------------------------------------------------------------------------
    def _figure_set_geometry_Windows( self, p_figure : Figure, p_xpos, p_ypos, p_width, p_height ):
        self._figure_set_geometry_default( p_figure = p_figure, p_xpos = p_xpos, p_ypos = p_ypos, p_width = p_width, p_height = p_height)


## -------------------------------------------------------------------------------------------------
    def _figure_set_geometry_Darwin( self, p_figure : Figure, p_xpos, p_ypos, p_width, p_height ):
        self._figure_set_geometry_default( p_figure = p_figure, p_xpos = p_xpos, p_ypos = p_ypos, p_width = p_width, p_height = p_height)


## -------------------------------------------------------------------------------------------------
    def _figure_set_geometry_default( self, p_figure : Figure, p_xpos, p_ypos, p_width, p_height ):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def figure_set_geometry( self, p_figure : Figure, p_xpos, p_ypos, p_width, p_height ):
        pass