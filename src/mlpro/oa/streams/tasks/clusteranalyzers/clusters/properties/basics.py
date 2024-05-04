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

from mlpro.bf.data.properties import *
from mlpro.bf.plot import Plottable
from mlpro.bf.math.normalizers import Renormalizable



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterProperty (Property, Plottable, Renormalizable):

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_derivative_order_max: DerivativeOrderMax = 0, p_visualize : bool = False ):
        Property.__init__(self, p_derivative_order_max = p_derivative_order_max)
        Plottable.__init__(self, p_visualize=p_visualize)

