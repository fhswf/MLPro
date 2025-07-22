## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.drifts
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-02-12  0.1.0     DA       Creation
## -- 2025-02-13  0.2.0     DA       Class Drift: new attributes event_id, drift_status
## -- 2025-02-19  0.3.0     DA       Class Drift: new parent Renomalizable
## -- 2025-05-28  0.4.0     DA/DS    Class Drift: new parent Change
## -- 2025-07-18  0.5.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.5.0 (2025-07-18)

This module provides a template class for types of data drift to be used in drift detection algorithms.
"""


from mlpro.oa.streams.tasks.changedetectors import Change



# Export list for public API
__all__ = [ 'Drift' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Drift (Change):
    """
    This is the base class for drift events raised by the drift detectors. See parent class 
    Change for more details.
    """

    pass


