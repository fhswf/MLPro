## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies
## -- Module  : clusterbased.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2024-02-25)
This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

# from matplotlib.figure import Figure
# from mlpro.bf.plot import PlotSettings
# from mlpro.oa.streams.basics import *
# from mlpro.oa.streams.basics import Instance, List
# import numpy as np
# from matplotlib.text import Text
# import matplotlib.patches as patches

from mlpro.oa.streams.tasks.anomalydetectors.anomalies.driftanomaly import DriftEvent


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftEventCB (DriftEvent):
    """
    Event class to be raised when cluster drift is detected.
    
    """

    C_NAME      = 'Cluster based Drift'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time : str, **p_kwargs):
        super().__init__(p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)
