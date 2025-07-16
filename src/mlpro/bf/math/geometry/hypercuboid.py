## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.math.geometry
## -- Module  : hypercuboid.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-05-29  0.0.0     DA       Creation
## -- 2024-06-03  1.0.0     DA       First implementation
## -- 2024-06-05  1.0.1     DA       Stabilization of Hypercuboid.set()
## -- 2024-06-26  1.1.0     DA       Refactoring of attribute color
## -- 2024-06-30  1.2.0     DA       Refactoring of method Hypercuboid.set()
## -- 2024-07-13  1.3.0     DA       Refactoring
## -- 2024-08-20  1.4.0     DA       New method Hypercuboid.check_collision()
## -- 2024-12-11  1.4.1     DA       Pseudo classes if matplotlib is not installed
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.1 (2024-12-11)

This module provides a property class for the geometric shape 'hypercuboid'.


""" 

from typing import Union
from datetime import datetime

import numpy as np

try:
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3D
except:
    class Figure : pass
    class Rectangle : pass
    class Poly3DCollection : pass
    class Line3D : pass

from mlpro.bf.plot import *
from mlpro.bf.math.properties import *
from mlpro.bf.math.normalizers import Normalizer
from mlpro.bf.math.geometry.basics import cprop_size_geo
from mlpro.bf.math.geometry.point import cprop_center_geo



# Export list for public API
__all__ = [ 'Hypercuboid',
            'cprop_hypercuboid',
            'cprop_hypercuboid_prev' ]




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
    color : str
        Plot color.
    alpha : float
        Alpha value of the plot to control the transparency of the cuboid.
    fill : bool
        If True, the cuboid is plotted with a filled surface.
    linewidth : float
        Width of the border lines of the cuboid.
    """

    C_PROPERTIES        = [ cprop_center_geo, cprop_size_geo ]

    C_PLOT_ACTIVE       = True
    C_PLOT_STANDALONE   = False
    C_PLOT_VALID_VIEWS  = [ PlotSettings.C_VIEW_2D, PlotSettings.C_VIEW_3D, PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW = PlotSettings.C_VIEW_ND

    C_PLOT_COLOR        = 'blue'
    C_PLOT_ALPHA        = 0.05
    C_PLOT_FILL         = True
    C_PLOT_LINEWIDTH    = 1
                  
## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name : str,
                  p_derivative_order_max: int = 0, 
                  p_value_prev : ValuePrev = False,
                  p_properties : PropertyDefinitions = [],
                  p_visualize: bool = False,
                  **p_kwargs ):
        
        super().__init__( p_name = p_name,
                          p_derivative_order_max = p_derivative_order_max,
                          p_value_prev = p_value_prev,
                          p_properties = p_properties,
                          p_visualize = p_visualize,
                          **p_kwargs )

        self.alpha      = self.C_PLOT_ALPHA
        self.fill       = self.C_PLOT_FILL
        self.linewidth  = self.C_PLOT_LINEWIDTH


## -------------------------------------------------------------------------------------------------
    def _get(self):
        return super()._get()

        
## -------------------------------------------------------------------------------------------------
    def set( self, 
             p_value, 
             p_time_stamp : Union[datetime, int, float] = None,
             p_upd_time_stamp : bool = True,
             p_upd_derivatives : bool = True ):         
            
        super().set( p_value = p_value, 
                     p_time_stamp = p_time_stamp,
                     p_upd_time_stamp = p_upd_time_stamp,
                     p_upd_derivatives = p_upd_derivatives )

        if p_value is None:
            self.center_geo.value = None
            self.size_geo.value   = 0
    
        else:
            val_np = np.array( self.value )

            self.center_geo.set( p_value = val_np.mean(axis=1),
                                 p_time_stamp = p_time_stamp )
        
            self.size_geo.set( p_value = np.prod( np.diff( val_np, axis=1 ) ), 
                               p_time_stamp = p_time_stamp )


## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure:Figure, p_settings:PlotSettings):
        super()._init_plot_2d( p_figure = p_figure, p_settings = p_settings )

        self._plot_2d_rectangle : Rectangle = None
        self._plot_line1 = None
        self._plot_line2 = None


## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure:Figure, p_settings:PlotSettings):
        super()._init_plot_3d( p_figure = p_figure, p_settings = p_settings )

        self._plot_3d_polycollection : Poly3DCollection = None
        self._plot_line1 : Line3D = None
        self._plot_line2 : Line3D = None
        self._plot_line3 : Line3D = None

    
## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure:Figure, p_settings:PlotSettings):
        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings: PlotSettings, **p_kwargs) -> bool:
        
        # 1 Intro
        if self.value is None: return False
        boundaries = self.value
        center_geo = self.center_geo.value 
        

        # 2 Determine the color of the cuboid
        if self.color is None: self.color = self.C_PLOT_COLOR
        color = self.color
        self.center_geo.color = self.color


        if self._plot_2d_rectangle is None:
            # 3 Init all plot objects
                                                 
            # 3.1 Init 2d rectangle
            self._plot_2d_rectangle = Rectangle( xy = ( boundaries[0][0], boundaries[1][0] ),
                                                 width = boundaries[0][1] - boundaries[0][0],
                                                 height = boundaries[1][1] - boundaries[1][0],
                                                 fill = self.fill,
                                                 edgecolor = color,
                                                 color = color,
                                                 facecolor = color,
                                                 linewidth = self.linewidth,
                                                 visible = True,
                                                 alpha = self.alpha )
            
            p_settings.axes.add_patch(self._plot_2d_rectangle)

            # 3.2 Init 2d crosshair through the geometric center
            self._plot_line1 = p_settings.axes.plot( [ center_geo[0], center_geo[0] ], 
                                                     [ boundaries[1][0], boundaries[1][1] ], 
                                                     color = color, 
                                                     linestyle = 'dashed', 
                                                     lw = 0.5 )[0]
            self._plot_line2 = p_settings.axes.plot( [ boundaries[0][0], boundaries[0][1] ], 
                                                     [ center_geo[1], center_geo[1] ], 
                                                     color = color, 
                                                     linestyle = 'dashed', 
                                                     lw = 0.5 )[0]            

        else:
            # 3 Update all plot objects
        
            # 3.1 Update 2d rectangle
            self._plot_2d_rectangle.set( xy = (boundaries[0][0], boundaries[1][0] ),
                                         width = boundaries[0][1] - boundaries[0][0],
                                         height = boundaries[1][1] - boundaries[1][0] )
            
            self._plot_2d_rectangle.set( edgecolor = color,
                                         facecolor = color,
                                         color = color )
                                                             
            # 3.2 Update crosshair lines
            self._plot_line1.set_data( [ center_geo[0], center_geo[0] ], 
                                       [ boundaries[1][0], boundaries[1][1] ] )
            self._plot_line2.set_data( [ boundaries[0][0], boundaries[0][1] ], 
                                       [ center_geo[1], center_geo[1] ] )  

            self._plot_line1.set( color = color )
            self._plot_line2.set( color = color )  

        return True                                                         
                                                         
                             
## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings: PlotSettings, **p_kwargs) -> bool:

        # 1 Intro
        if self.value is None: return False
        b = self.value
        center_geo = self.center_geo.value 


        # 2 Determine the color of the cuboid
        if self.color is None: self.color = self.C_PLOT_COLOR
        color = self.color
        self.center_geo.color = self.color
            
        
        if self._plot_3d_polycollection is None:
            # 3 Init all plot objects
                                                             
            # 3.1 Init 3d cuboid
            self._plot_3d_polycollection = Poly3DCollection( verts= [], 
                                                             linewidths=self.linewidth,
                                                             edgecolors=color, 
                                                             facecolors=color, 
                                                             alpha = self.alpha )
                             
            self._plot_settings.axes.add_collection(self._plot_3d_polycollection)

            # 2.2 Init 3d crosshair 
            self._plot_line1 = p_settings.axes.plot( [ center_geo[0], center_geo[0] ], 
                                                     [ center_geo[1], center_geo[1] ], 
                                                     [ b[2][0], b[2][1] ], 
                                                     color = color, 
                                                     linestyle = 'dashed', 
                                                     lw = 0.5 )[0]
            self._plot_line2 = p_settings.axes.plot( [ center_geo[0], center_geo[0] ], 
                                                     [ b[1][0], b[1][1] ],
                                                     [ center_geo[2], center_geo[2] ], 
                                                     color = color, 
                                                     linestyle = 'dashed', 
                                                     lw = 0.5 )[0]
            self._plot_line3 = p_settings.axes.plot( [ b[0][0], b[0][1] ], 
                                                     [ center_geo[1], center_geo[1] ], 
                                                     [ center_geo[2], center_geo[2] ], 
                                                     color = color, 
                                                     linestyle = 'dashed', 
                                                     lw = 0.5 )[0]
        
        else:
            # 3 Update the 3d crosshair
            self._plot_line1.set_data_3d( [ center_geo[0], center_geo[0] ], 
                                          [ center_geo[1], center_geo[1] ], 
                                          [ b[2][0], b[2][1] ] )
            self._plot_line2.set_data_3d( [ center_geo[0], center_geo[0] ], 
                                          [ b[1][0], b[1][1] ],
                                          [ center_geo[2], center_geo[2] ] )
            self._plot_line3.set_data_3d( [ b[0][0], b[0][1] ], 
                                          [ center_geo[1], center_geo[1] ], 
                                          [ center_geo[2], center_geo[2] ] )
            
            self._plot_line1.set( color = color )
            self._plot_line2.set( color = color )
            self._plot_line3.set( color = color )


        # 4 Update the 3d cuboid
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

        self._plot_3d_polycollection.set( edgecolor = color,
                                          facecolor = color )
        
        return True


## -------------------------------------------------------------------------------------------------
    def _remove_plot_2d(self):
        if self._plot_2d_rectangle is None: return
    
        self._plot_2d_rectangle.remove()
        self._plot_2d_rectangle = None
        self._plot_line1.remove()
        self._plot_line1 = None
        self._plot_line2.remove()
        self._plot_line2 = None


## -------------------------------------------------------------------------------------------------
    def _remove_plot_3d(self):
        if self._plot_3d_polycollection is None: return
    
        self._plot_3d_polycollection.remove()
        self._plot_3d_polycollection = None
        self._plot_line1.remove()
        self._plot_line1 = None
        self._plot_line2.remove()
        self._plot_line2 = None
        self._plot_line3.remove()
        self._plot_line3 = None


## -------------------------------------------------------------------------------------------------
    def renormalize(self, p_normalizer: Normalizer):
        super().renormalize( p_normalizer = p_normalizer)

        self._value = p_normalizer.renormalize( p_data=np.array(self.value) )


## -------------------------------------------------------------------------------------------------
    def check_collision(self, p_hypercuboid ) -> bool:
        bdr_other  = p_hypercuboid.value
        collision  = True

        for dim, (bdr_left,bdr_right) in enumerate(self._value):
            if ( bdr_other[dim][0] > bdr_right ) or ( bdr_other[dim][1] < bdr_left ):
                collision = False
                break

        return collision


## -------------------------------------------------------------------------------------------------
    value       = property( fget = _get, fset = set)






cprop_hypercuboid      : PropertyDefinition = ( 'hypercuboid', 0, False, Hypercuboid )
cprop_hypercuboid_prev : PropertyDefinition = ( 'hypercuboid', 0, True, Hypercuboid )