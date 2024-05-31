## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.math.geometry
## -- Module  : hypercuboid.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-05-29  0.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2024-05-29)

This module provides classes for hypercuboids.

""" 

from matplotlib.patches import Rectangle

from mlpro.bf.plot import *
from mlpro.bf.math.properties import *
from mlpro.bf.math.normalizers import Normalizer
from mlpro.bf.math.geometry import Point, cprop_center_geo




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Hypercuboid (MultiProperty):
    """
    Implementation of a point in a hyper space. Current position, velocity and acceleration are managed.

    Attributes
    ----------
    values
        Current boundaries of the hypercuboid as two-dimensional array-like data object. For a
        n-dimensional hypercuboid value[d][0] determines the lower boundary in dimension d while
        value[d][1] specifies the upper boundary.
    """

    C_PROPERTIES        = [ cprop_center_geo ]

    C_PLOT_ACTIVE       = True
    C_PLOT_STANDALONE   = False
    C_PLOT_VALID_VIEWS  = [ PlotSettings.C_VIEW_2D, PlotSettings.C_VIEW_3D, PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure:Figure, p_settings:PlotSettings):
        self._plot_2d_rectangle : Rectangle = None


## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure:Figure, p_settings:PlotSettings):
        pass


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure:Figure, p_settings:PlotSettings):
        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs):
        
        if self._plot_2d_rectangle is None:
            self._plot_2d_rectangle = Rectangle( xy = (self.values[0][0], self.values[1][0] ),
                                                 width = self.values[0][1] - self.values[0][0],
                                                 height = self.values[0][1] - self.values[0][0],
                                                 fill = True,
                                                 edgecolor = 'red',
                                                 color = 'red',
                                                 alpha = 0.5 )
    
        else:
            self._plot_2d_rectangle.set( xy = (self.values[0][0], self.values[1][0] ),
                                         width = self.values[0][1] - self.values[0][0],
                                         height = self.values[0][1] - self.values[0][0] )
    
                                                         
## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings: PlotSettings, **p_kwargs):
        pass
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings: PlotSettings, **p_kwargs):
        pass


## -------------------------------------------------------------------------------------------------
    def _remove_plot_2d(self):
        pass


## -------------------------------------------------------------------------------------------------
    def _remove_plot_3d(self):
        pass


## -------------------------------------------------------------------------------------------------
    def _remove_plot_nd(self):
        pass        


## -------------------------------------------------------------------------------------------------
    def renormalize(self, p_normalizer: Normalizer):
        raise NotImplementedError





cprop_hypercuboid : PropertyDefinition = ( 'hypercuboid', 0, False, Hypercuboid )
