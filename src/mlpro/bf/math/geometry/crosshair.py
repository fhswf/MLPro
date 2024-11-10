## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.math.geometry
## -- Module  : crosshair.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-10-31  1.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-10-31)

This module provides the class Crosshair that provides crosshair functionality.

"""

from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3D

from mlpro.bf.plot import PlotSettings
from mlpro.bf.streams import *
from mlpro.bf.various import Id
from mlpro.bf.math.geometry import Point
from mlpro.bf.math.properties import *




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

    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False
    C_PLOT_VALID_VIEWS      = [ PlotSettings.C_VIEW_2D, 
                                PlotSettings.C_VIEW_3D, 
                                PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW     = PlotSettings.C_VIEW_ND
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
    def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs):

        # 0 Intro
        if self.value is None: return

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


## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings: PlotSettings, **p_kwargs):

        # 0 Intro
        if self.value is None: return

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


        # 3 Plot a crosshair with label texts
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




cprop_crosshair      : PropertyDefinition = ( 'crosshair', 0, False, Crosshair )
