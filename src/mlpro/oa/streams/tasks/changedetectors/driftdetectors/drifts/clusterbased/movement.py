## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.drifts.clusterbased
## -- Module  : movement.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-02-13  1.0.0     DA       Creation
## -- 2025-07-18  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-18) 

This module provides a sub-type of class DriftCB related to cluster movement.
"""

from mlpro.oa.streams.tasks.changedetectors.driftdetectors.drifts.clusterbased.basics import DriftCB



# Export list for public API
__all__ = [ 'DriftCBMovement' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftCBMovement (DriftCB):
    """
    Sub-type indicating the begin or end of a cluster movement.
    """

    C_PLOT_ACTIVE   = True