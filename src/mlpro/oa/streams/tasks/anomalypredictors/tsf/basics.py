## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.anomalypredictors
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-06-04  0.0.0     DA/DS    Creation
## -- 2024-08-23  0.1.0     DA/DS    Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2024-08-23)

This module provides basic templates for online anomaly prediction in MLPro.
 
"""


from mlpro.bf.math import Function
from mlpro.oa.streams.tasks.anomalypredictors import AnomalyPredictor, AnomalyPrediction
from mlpro.sl import SLAdaptiveFunction



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TimeSeriesForcaster (Function):
    """
    ...
    """

    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OATimeSeriesForcaster (TimeSeriesForcaster, SLAdaptiveFunction):
    """
    ...
    """

    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyPredictorTSF (AnomalyPredictor):
    """
    ...
    """
    
    pass
