## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies.clusterbased
## -- Module  : deformation.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-02-11  0.0.0     DS       Creation
## -- 2025-07-18  0.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-07-18)

This module provides a template class for cluster deformation to be used in anomaly detection algorithms.
"""

from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.anomalies.clusterbased.basics import AnomalyCB



# Export list for public API
__all__ = [ 'ClusterDeformation' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterDeformation (AnomalyCB):
    """
    Event class to be raised when a cluster deforms.
    
    """

    pass



