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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
    def __init__( self, 
                  p_name : str,
                  p_derivative_order_max: int = 0, 
                  p_value_prev : ValuePrev = False,
                  p_properties : PropertyDefinitions = [],
                  p_visualize: bool = False ):
        
        super().__init__( p_name = p_name,
                          p_derivative_order_max = p_derivative_order_max,
                          p_value_prev = p_value_prev,
                          p_properties = p_properties,
                          p_visualize = p_visualize )

        self.color = 'blue'
        self.alpha = 0.1
        self.fill  = True


## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure:Figure, p_settings:PlotSettings):
        self._plot_2d_rectangle : Rectangle = None


## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure:Figure, p_settings:PlotSettings):
        self._plot_3d_polycollection : Poly3DCollection = None


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure:Figure, p_settings:PlotSettings):
        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs):
        
        if self.value is None: return

        if self._plot_2d_rectangle is None:
            self._plot_2d_rectangle = Rectangle( xy = (self.value[0][0], self.value[1][0] ),
                                                 width = self.value[0][1] - self.value[0][0],
                                                 height = self.value[1][1] - self.value[1][0],
                                                 fill = self.fill,
                                                 edgecolor = self.color,
                                                 color = self.color,
                                                 facecolor = self.color,
                                                 visible = True,
                                                 alpha = self.alpha )
            
            p_settings.axes.add_patch(self._plot_2d_rectangle)
    
        else:
            self._plot_2d_rectangle.set( xy = (self.value[0][0], self.value[1][0] ),
                                         width = self.value[0][1] - self.value[0][0],
                                         height = self.value[1][1] - self.value[1][0] )
    
                                                         
## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings: PlotSettings, **p_kwargs):

        # 1 Intro
        if self.value is None: return

        
        # 2 Initialization of the cuboid
        if self._plot_3d_polycollection is None:
            self._plot_3d_polycollection = Poly3DCollection( verts= [], 
                                                             edgecolors=self.color, 
                                                             facecolors=self.color, 
                                                             alpha = self.alpha )
                             
            self._plot_settings.axes.add_collection(self._plot_3d_polycollection)


        # 3 Update of the cuboid
        b = self.value

        verts = np.asarray([[[b[0][0], b[1][0], b[2][1]],
                             [b[0][1], b[1][0], b[2][1]],
                             [b[0][1], b[1][0], b[2][0]],
                             [b[0][0], b[1][0], b[2][0]]],

                            [[b[0][0], b[1][0], b[2][1]],
                             [b[0][1], b[1][0], b[2][1]],
                             [b[0][1], b[1][1], b[2][1]],
                             [b[0][0], b[1][1], b[2][1]]],

                            [[b[0][0], b[1][0], b[2][1]],
                             [b[0][0], b[1][1], b[2][1]],
                             [b[0][0], b[1][1], b[2][0]],
                             [b[0][0], b[1][0], b[2][0]]],

                            [[b[0][1], b[1][0], b[2][1]],
                             [b[0][1], b[1][1], b[2][1]],
                             [b[0][1], b[1][1], b[2][0]],
                             [b[0][1], b[1][0], b[2][0]]],

                            [[b[0][0], b[1][1], b[2][1]],
                             [b[0][1], b[1][1], b[2][1]],
                             [b[0][1], b[1][1], b[2][0]],
                             [b[0][0], b[1][1], b[2][0]]],

                            [[b[0][0], b[1][0], b[2][0]],
                             [b[0][1], b[1][0], b[2][0]],
                             [b[0][1], b[1][1], b[2][0]],
                             [b[0][0], b[1][1], b[2][0]]]])

        self._plot_3d_polycollection.set_verts(verts)
   

## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings: PlotSettings, **p_kwargs):
        if self.value is None: return
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
