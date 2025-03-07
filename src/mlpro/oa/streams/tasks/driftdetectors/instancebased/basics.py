## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.streams.tasks.driftdetectors.instancebased
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-04  0.1.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-03-04)

This module provides a template for instance-based drift detection algorithms to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.tasks.driftdetectors.basics import DriftDetector




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftDetectorIB (DriftDetector):
    """
    This is the base class for online-adaptive instance-based drift detectors. It raises an event 
    when a drift is detected.
    """

    C_TYPE = 'Instance-based Drift Detector'

       