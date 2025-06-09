## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.drifts.clusterbased
## -- Module  : geometric_center.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-06-08  1.0.0     DS       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-06-08)

This module provides a sub-typ of class DriftCB related to geometric center of a cluster.
"""

from mlpro.oa.streams.tasks.changedetectors.driftdetectors.drifts.clusterbased.basics import DriftCB




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftCBCenterGeo (DriftCB):
    """
    Sub-type indicating the begin or end of a geometric center of a cluster.
    """

    C_PLOT_ACTIVE   = True