## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies
## -- Module  : driftanomaly.py
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


from mlpro.bf.plot import PlotSettings
from mlpro.bf.streams import Instance
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.basics import Anomaly
from matplotlib.figure import Figure
from matplotlib.text import Text



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftAnomaly (Anomaly):
    """
    Event class to be raised when drift is detected.
    
    """

    C_NAME      = 'Drift'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time : str, p_magnitude : float, p_rate : float, **p_kwargs):
        super().__init__(p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)
