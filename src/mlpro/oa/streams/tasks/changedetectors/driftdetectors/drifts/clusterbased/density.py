## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.drifts.clusterbased
## -- Module  : density.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-06-08  1.0.0     DS       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-06-08)

This module provides a sub-typ of class DriftCB related to density of a cluster.
"""

from mlpro.oa.streams.tasks.changedetectors.driftdetectors.drifts.clusterbased.basics import DriftCB




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftCBDensity (DriftCB):
    """
    Sub-type indicating the begin or end of a density of a cluster.
    """

    C_PLOT_ACTIVE   = True