## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.math.geometry
## -- Module  : crosshair.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-31  1.0.0     DA       Creation
## -- 2024-12-11  1.0.1     DA       Pseudo classes if matplotlib is not installed
## -- 2025-06-09  1.1.0     DA       Refactoring of Crosshair._update_plot*: new return parameter
## -- 2025-06-25  1.2.0     DA       Class Crosshair: implementation of nD plot methods
## -- 2025-06-26  1.2.1     DA       Bugfix in method Crosshair._update_plot_nd()
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.1 (2025-06-26)

This module provides the class Crosshair that provides crosshair functionality.

"""

try:
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Line3D
except:
    class Figure : pass
    class Line3D : pass

from mlpro.bf.plot import PlotSettings
from mlpro.bf.streams import *
from mlpro.bf.math.geometry import Point
from mlpro.bf.math.properties import *



# Export list for public API
__all__ = [ 'Crosshair',
            'cprop_crosshair' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Crosshair (Point): 
    """
    This managed property provides a crosshair functionality including 
    - managing its position
    - optionally its velocity and acceleration as auto-derivatives
    - plot functionality
    - renormalization

    Parameters
    ----------
    p_name : str
        Name of the property
    p_derivative_order_max : DerivativeOrderMax
        Maximum order of auto-generated derivatives (numeric properties only).
    p_value_prev : bool
        If True, the previous value is stored in value_prev whenever value is updated.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_kwargs : dict
        Keyword parameters.
    """

    C_PLOT_COLOR            = 'blue'

## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        super()._init_plot_2d(p_figure=p_figure, p_settings=p_settings)
        self._plot_line1 = None
        self._plot_line2 = None
    

## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure: Figure, p_settings: PlotSettings):
        super()._init_plot_3d(p_figure=p_figure, p_settings=p_settings)
        self._plot_line1 : Line3D = None
        self._plot_line2 : Line3D = None
        self._plot_line3 : Line3D = None


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure, p_settings):
        super()._init_plot_nd(p_figure, p_settings)
        self._plot_lines = []
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs) -> bool:

        # 0 Intro
        if self.value is None: return False

        if self.color is None:
            self.color = self.C_PLOT_COLOR


        # 1 Plot the point 
        color = self.color
        self.color = Point.C_PLOT_COLOR
        super()._update_plot_2d(p_settings, **p_kwargs)
        self.color = color


        # 2 Get line coordinates
        center   = self.value
        ax_xlim  = p_settings.axes.get_xlim()
        ax_ylim  = p_settings.axes.get_ylim()
        xlim     = [ min( ax_xlim[0], center[0] ), max(ax_xlim[1], center[0] ) ]
        ylim     = [ min( ax_ylim[0], center[1] ), max(ax_ylim[1], center[1] ) ]

            
        # 3 Plot a crosshair
        if self._plot_line1 is None:
            # 3.1 Add initial crosshair lines
            self._plot_line1 = p_settings.axes.plot( xlim, [center[1],center[1]], color=self.color, linestyle='dashed', lw=1)[0]
            self._plot_line2 = p_settings.axes.plot( [center[0],center[0]], ylim, color=self.color, linestyle='dashed', lw=1)[0]
        else:
            # 3.2 Update data of crosshair lines
            self._plot_line1.set_data( xlim, [center[1],center[1]] )
            self._plot_line1.set_color( self.color )
            self._plot_line2.set_data( [center[0],center[0]], ylim )
            self._plot_line2.set_color( self.color )

        return True


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings: PlotSettings, **p_kwargs) -> bool:

        # 0 Intro
        if self.value is None: return False

        if self.color is None:
            self.color = self.C_PLOT_COLOR


        # 1 Plot the point 
        color = self.color
        self.color = Point.C_PLOT_COLOR
        super()._update_plot_3d(p_settings, **p_kwargs)
        self.color = color

        
        # 2 Get coordinates
        center   = self.value
        ax_xlim  = p_settings.axes.get_xlim()
        ax_ylim  = p_settings.axes.get_ylim()
        ax_zlim  = p_settings.axes.get_zlim()
        xlim     = [ min( ax_xlim[0], center[0] ), max(ax_xlim[1], center[0] ) ]
        ylim     = [ min( ax_ylim[0], center[1] ), max(ax_ylim[1], center[1] ) ]
        zlim     = [ min( ax_zlim[0], center[2] ), max(ax_zlim[1], center[2] ) ]


        # 3 Plot a crosshair
        if self._plot_line1 is None:
            # 3.1 Add initial crosshair lines
            self._plot_line1 = p_settings.axes.plot( xlim, [center[1],center[1]], [center[2],center[2]], color=self.color, linestyle='dashed', lw=1)[0]
            self._plot_line2 = p_settings.axes.plot( [center[0],center[0]], ylim, [center[2],center[2]], color=self.color, linestyle='dashed', lw=1)[0]
            self._plot_line3 = p_settings.axes.plot( [center[0],center[0]], [center[1],center[1]], zlim, color=self.color, linestyle='dashed', lw=1)[0]

        else:
            # 3.2 Update data of crosshair lines
            self._plot_line1.set_data_3d( xlim, [center[1],center[1]], [center[2],center[2]] )
            self._plot_line1.set_color( self.color )
            self._plot_line2.set_data_3d( [center[0],center[0]], ylim, [center[2],center[2]] )
            self._plot_line2.set_color( self.color )
            self._plot_line3.set_data_3d( [center[0],center[0]], [center[1],center[1]], zlim )
            self._plot_line3.set_color( self.color )

        return True
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings : PlotSettings, **p_kwargs) -> bool: 

        # 0 Intro
        if self.value is None: return False

        if self.color is None:
            self.color = self.C_PLOT_COLOR


        # 2 Get coordinates
        center   = self.value
        xlim  = p_settings.axes.get_xlim()
        #xlim     = [ min( ax_xlim[0], center[0] ), max(ax_xlim[1], center[0] ) ]
        

        # 3 Plot a crosshair per feature
        if not self._plot_lines:
            # 3.1 Add initial crosshair lines
            for center_fval in center: 
                self._plot_lines.append( p_settings.axes.plot( xlim, [center_fval,center_fval], color=self.color, linestyle='dashed', lw=1)[0] )

        else:
            # 3.2 Update data of crosshair lines
            for i, plot_line in enumerate(self._plot_lines):
                plot_line.set_data( xlim, [center[i],center[i]] )
                plot_line.set_color( self.color )

        return True
    

## -------------------------------------------------------------------------------------------------
    def _remove_plot_2d(self):
        super()._remove_plot_2d()

        if self._plot_line1 is None: return

        self._plot_line1.remove()
        self._plot_line1 = None

        self._plot_line2.remove()
        self._plot_line2 = None


## -------------------------------------------------------------------------------------------------
    def _remove_plot_3d(self):
        super()._remove_plot_3d()

        if self._plot_line1 is None: return
        
        self._plot_line1.remove()
        self._plot_line1 = None

        self._plot_line2.remove()
        self._plot_line2 = None

        self._plot_line3.remove()
        self._plot_line3 = None


## -------------------------------------------------------------------------------------------------
    def _remove_plot_nd(self):
        for plot_line in self._plot_lines:
            plot_line.remove()

        self._plot_lines.clear()



cprop_crosshair      : PropertyDefinition = ( 'crosshair', 0, False, Crosshair )
