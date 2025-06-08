## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.drifts.clusterbased
## -- Module  : size.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-06-06  1.0.0     DS       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-06-06)

This module provides a sub-typ of class DriftCB related to cluster size.
"""

from mlpro.oa.streams.tasks.changedetectors.driftdetectors.drifts.clusterbased.basics import DriftCB




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftCBSize (DriftCB):
    """
    Sub-type indicating the begin or end of a cluster size.
    """

    C_PLOT_ACTIVE   = True