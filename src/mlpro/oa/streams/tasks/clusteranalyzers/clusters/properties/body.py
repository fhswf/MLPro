## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties
## -- Module  : centroid.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-05-04  0.1.0     DA       Creation
## -- 2024-05-29  0.2.0     DA       Refactoring
## -- 2024-05-30  0.3.0     DA       Global aliases: new boolean param ValuePrev
## -- 2024-05-31  0.4.0     DA       Improved the stability of the plot methods
## -- 2024-06-13  0.5.0     DA       New property definitions cprop_centroid_prev*
## -- 2024-06-26  0.6.0     DA       Refactoring
## -- 2024-07-13  0.7.0     DA       Refactoring
## -- 2024-10-31  0.8.0     DA       New parent class Crosshair
## -- 2024-12-11  0.8.1     DA       Pseudo classes if matplotlib is not installed
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.8.1 (2024-12-11)

This module provides ...

"""

try:
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from mpl_toolkits.mplot3d.art3d import Line3D, Text3D
except:
    class Figure : pass
    class Text : pass
    class Line3D : pass
    class Text3D : pass

from mlpro.bf.mt import Figure, PlotSettings
from mlpro.bf.various import *
from mlpro.bf.plot import *
from mlpro.bf.streams import *
from mlpro.bf.various import Id
#from mlpro.bf.math.geometry import Crosshair
from mlpro.bf.math.properties import *
from mlpro.oa.streams.tasks.clusteranalyzers.clusters import Cluster




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Body (MultiProperty, Id): 
    """
    This managed property provides full centroid functionality including 
    - managing its position
    - optionally its velocity and acceleration as auto-derivatives
    - plot functionality
    - renormalization

    Hint: please assign the id of the cluster to the centroid as well to get a proper visualization.

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

    C_PROPERTIES : PropertyDefinitions = []

    C_PLOT_ACTIVE                      = True
    C_PLOT_STANDALONE                  = False
    C_PLOT_VALID_VIEWS                 = [ PlotSettings.C_VIEW_2D, 
                                           PlotSettings.C_VIEW_3D, 
                                           PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW                = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name : str,
                  p_derivative_order_max: int = 0, 
                  p_value_prev : ValuePrev = False,
                  p_properties : PropertyDefinitions = [],
                  p_visualize: bool = False,
                  **p_kwargs ):

        MultiProperty.__init__( self, 
                                p_name = p_name, 
                                p_derivative_order_max = p_derivative_order_max,
                                p_value_prev = p_value_prev, 
                                p_properties = p_properties,
                                p_visualize = p_visualize,
                                **p_kwargs )
        
        Id.__init__( self, p_id = 0 )

        self.color = None
  




# Centroid with 0,1,2 order derivatives and plot functionality with/without storing previous values
cprop_centroid       : PropertyDefinition = ( 'centroid', 0, False, Centroid )
cprop_centroid1      : PropertyDefinition = ( 'centroid', 1, False, Centroid )
cprop_centroid2      : PropertyDefinition = ( 'centroid', 2, False, Centroid )

cprop_centroid_prev  : PropertyDefinition = ( 'centroid', 0, True, Centroid )
cprop_centroid_prev1 : PropertyDefinition = ( 'centroid', 1, True, Centroid )
cprop_centroid_prev2 : PropertyDefinition = ( 'centroid', 2, True, Centroid )

# Geometric center with 0,1,2 order derivatives and plot functionality
cprop_center_geo     : PropertyDefinition = ( 'center_geo', 0, False, Centroid )
cprop_center_geo1    : PropertyDefinition = ( 'center_geo', 1, False, Centroid )
cprop_center_geo2    : PropertyDefinition = ( 'center_geo', 2, False, Centroid )