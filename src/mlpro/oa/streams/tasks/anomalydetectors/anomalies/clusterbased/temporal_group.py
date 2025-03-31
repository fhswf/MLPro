## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies.clusterbased
## -- Module  : temporal_group.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-31  0.0.0     DS       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2025-03-31)

This module provides a class for temporal group anomalies to be used in anomaly detection algorithms.
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
class TemporalGroupAnomaly (AnomalyCB):
    """
    Event class for anomaly events when temporal group anomalies are detected.
    
    Parameters
    ----------
    p_instances : Instance
        List of instances. Default value = None.
    p_id : int
        Anomaly ID. Default value = 0.
    p_tstamp : datetime = None
        Time of occurance of anomaly. Default = None.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_raising_object : object
        Reference of the object raised. Default = None.
    p_mean : float
        The mean value of the anomaly. Default = None.
    p_mean_deviation : float
        The mean deviation of the anomaly. Default = None.
    **p_kwargs
        Further optional keyword arguments.
    """

    pass

