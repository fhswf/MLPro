## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies.clusterbased
## -- Module  : group.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-11  0.0.0     DA       Creation
## -- 2025-07-18  0.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-07-18)

This module provides a class for group anomalies to be used in anomaly detection algorithms.
"""

from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.anomalies.clusterbased.basics import AnomalyCB



# Export list for public API
__all__ = [ 'GroupAnomaly', 
            'SpatialGroupAnomaly', 
            'TemporalGroupAnomaly' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GroupAnomaly (AnomalyCB):
    """
    Event class for anomaly events when group anomalies are detected.
    """

    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SpatialGroupAnomaly (GroupAnomaly):

    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TemporalGroupAnomaly (GroupAnomaly):

    pass