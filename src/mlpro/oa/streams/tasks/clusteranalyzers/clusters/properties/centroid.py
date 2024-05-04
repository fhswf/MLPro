## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.clusteranalyzers
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-05-04  0.1.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2024-05-04)

This module provides ...

"""

from mlpro.bf.data.properties import DerivativeOrderMax
from mlpro.bf.math.geometry import Point
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties.basics import ClusterProperty




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Centroid (ClusterProperty): 

    def __init__( self, p_derivative_order_max: DerivativeOrderMax = 0, p_visualize: bool = False):
        super().__init__(p_derivative_order_max, p_visualize)

        # ...
        


## -------------------------------------------------------------------------------------------------
    def _get_value(self):
        """
        Internal method to determine the dimensionality of the currently stored values. It is used
        implicitely when accessing attribute 'dim'.
        """

        pass
            

## -------------------------------------------------------------------------------------------------
    value = property( fget=_get_value )



