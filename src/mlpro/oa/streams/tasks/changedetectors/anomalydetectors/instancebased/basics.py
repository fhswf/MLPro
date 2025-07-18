## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.anomalydetectors.instancebased
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-02-28  1.0.0     DA       Creation
## -- 2025-07-18  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-18)

This module provides MLPro's template class for instance-based anomaly detectors.
"""


from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.basics import AnomalyDetector



# Export list for public API
__all__ = [ 'AnomalyDetectorIB' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorIB (AnomalyDetector):
    """
    This class is a sub-type template for instance-based anomaly detectors.
    """

    C_TYPE = 'Anomaly Detector (IB)'