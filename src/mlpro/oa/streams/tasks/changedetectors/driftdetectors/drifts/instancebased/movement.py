## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.drifts.instancebased
## -- Module  : movement.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-04  1.0.0     DA/DS    Creation
## -- 2025-07-18  1.1.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2025-07-18)

This module provides a sub-typ of class DriftIB related to data drift of type movement.
"""

from mlpro.oa.streams.tasks.changedetectors.driftdetectors.drifts.instancebased.basics import DriftIB



# Export list for public API
__all__ = [ 'DriftIBMovement' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftIBMovement (DriftIB):
    """
    Sub-type indicating the begin or end of a data drift of type movement.
    """

    pass
