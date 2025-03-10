## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.anomalydetectors.instancebased
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-02-28  1.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-02-28)

This module provides MLPro's template class for instance-based anomaly detectors.
"""


from mlpro.oa.streams.tasks.anomalydetectors.basics import AnomalyDetector



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorIB (AnomalyDetector):
    """
    This class is a sub-type template for instance-based anomaly detectors.
    """

    C_TYPE = 'Anomaly Detector (IB)'