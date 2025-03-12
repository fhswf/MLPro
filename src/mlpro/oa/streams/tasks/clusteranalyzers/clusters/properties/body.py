## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties
## -- Module  : body.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-12  0.1.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-03-12)

This module provides a template class for the cluster property 'body'.

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
from mlpro.bf.math.geometry import cprop_size_geo
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import cprop_center_geo




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Body (MultiProperty): 
    """
    ...

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

    C_PROPERTIES : PropertyDefinitions = [ cprop_center_geo,
                                           cprop_size_geo ]

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
  




# Property definitions for cluster bodies
cprop_body  : PropertyDefinition = ( 'body', 0, False, Body )
