## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.drifts.clusterbased
## -- Module  : size_geo.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-07-29  1.0.0     DS       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-07-29) 

This module provides a sub-type of class DriftCB related to geometric cluster size.
"""

from mlpro.oa.streams.tasks.changedetectors.driftdetectors.drifts.clusterbased.basics import DriftCB



# Export list for public API
__all__ = [ 'DriftCBSizeGeo' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftCBSizeGeo (DriftCB):
    """
    Sub-type indicating the begin or end of a cluster size.
    """

    C_PLOT_ACTIVE   = True