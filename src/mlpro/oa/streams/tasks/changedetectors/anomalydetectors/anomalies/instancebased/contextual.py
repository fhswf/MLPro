## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies.instancebased
## -- Module  : contextual.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -- 2025-02-28  1.3.0     DA       Refactoring
## -- 2025-07-18  1.4.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.0 (2025-07-18)

This module provides a template class for contextual anomaly to be used in anomaly detection algorithms.
"""

from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.anomalies.instancebased.basics import AnomalyIB



# Export list for public API
__all__ = [ 'ContextualAnomaly' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ContextualAnomaly (AnomalyIB):
    """
    Event class for contextual anomaly events.
    """

    pass