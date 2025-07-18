## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.plot.backends
## -- Module  : qtagg.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-12-30  0.1.0     DA       Initial implementation
## -- 2025-01-03  0.2.0     DA       Refactoring
## -- 2025-04-05  0.3.0     DA       Aligment with Qt6/PySide 6.9.0
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.3.0 (2025-04-05)

This module provides an integration for Matplotlib backend 'qtagg'.

"""


import re

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.qt_compat import QtCore
except:
    class Figure : pass

from mlpro.bf.plot.backends import PlotBackend, WindowGeometry, WindowState, WSNORMAL, WSMINIMIZED, WSMAXIMIZED



# Export list for public API
__all__ = [ 'PlotBackendqtagg',
            'PlotBackendQtAgg' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PlotBackendqtagg (PlotBackend):
    """
    Integrates the Matplotlib backend 'qtagg' into MLPro.
    """

    C_NAME      = 'qtagg'

## -------------------------------------------------------------------------------------------------
    def _figure_force_foreground_default(self, p_figure : Figure):
        window = p_figure.canvas.manager.window
        window.setWindowFlags(window.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
        window.show()


## -------------------------------------------------------------------------------------------------
    def _figure_get_geometry_default(self, p_figure : Figure) -> WindowGeometry:

        # 1 Get size and position
        window = p_figure.canvas.manager.window
        pos    = window.pos()
        size   = window.size()

        # 2 Get window state
        state_qt = window.windowState()
        if state_qt & QtCore.Qt.WindowState.WindowMinimized:
            state = WSMINIMIZED
        elif state_qt & QtCore.Qt.WindowState.WindowMaximized:
            state = WSMAXIMIZED
        else:
            state = WSNORMAL

        # 3 Return dictionary compatible with type WindowGeometry
        return { 'xpos'   : pos.x(),
                 'ypos'   : pos.y(),
                 'width'  : size.width(),
                 'height' : size.height(),
                 'state'  : state }
    

## -------------------------------------------------------------------------------------------------
    def _figure_set_geometry_default(self, p_figure : Figure, p_geometry : WindowGeometry):

        # 1 Set size and position
        window = p_figure.canvas.manager.window
        window.move( p_geometry['xpos'], p_geometry['ypos'] )
        window.resize( p_geometry['width'], p_geometry['height'] )

        # 2 Set window state
        state = p_geometry['state']
        if state == WSMINIMIZED:
            state_qt = QtCore.Qt.WindowState.WindowMinimized
        elif state == WSMAXIMIZED:
            state_qt = QtCore.Qt.WindowState.WindowMaximized
        else:
            state_qt = QtCore.Qt.WindowState.WindowNoState

        window.setWindowState(state_qt)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PlotBackendQtAgg (PlotBackendqtagg): pass