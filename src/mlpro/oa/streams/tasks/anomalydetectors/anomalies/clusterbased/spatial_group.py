## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies.clusterbased
## -- Module  : spatial_group.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-31  0.0.0     DS       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2025-03-31)

This module provides a class for spatial group anomalies to be used in anomaly detection algorithms.
"""

from datetime import datetime

try:
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from matplotlib import patches
except:
    class Figure : pass
    class Text : pass
    class patches : pass
    
from mlpro.bf.plot import PlotSettings
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased.basics import AnomalyCB



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SpatialGroupAnomaly (AnomalyCB):

    pass

