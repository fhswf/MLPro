## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.drifts.clusterbased
## -- Module  : movement.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-02-13  1.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-02-13)

This module provides a sub-typ of class DriftCB related to cluster movement.
"""

from mlpro.oa.streams.tasks.driftdetectors.drifts.clusterbased.basics import DriftCB




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftCBMovement (DriftCB):
    """
    Sub-type indicating the begin or end of a cluster movement.
    """

    C_PLOT_ACTIVE   = True