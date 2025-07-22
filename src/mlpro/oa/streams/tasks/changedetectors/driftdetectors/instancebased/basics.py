## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.driftdetectors.instancebased
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-04  0.1.0     DA       Creation
## -- 2025-07-18  0.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2025-07-18)

This module provides a template for instance-based drift detection algorithms to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.tasks.changedetectors.driftdetectors.basics import DriftDetector




# Export list for public API
__all__ = [ 'DriftDetectorIB' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftDetectorIB (DriftDetector):
    """
    This is the base class for online-adaptive instance-based drift detectors. It raises an event 
    when a drift is detected.
    """

    C_TYPE = 'Instance-based Drift Detector'