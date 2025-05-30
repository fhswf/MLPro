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
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.4.0 (2025-05-28)

This module provides a template class for types of data drift to be used in drift detection algorithms.
"""


from mlpro.oa.streams.tasks.changedetectors import Change



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Drift (Change):
    """
    This is the base class for drift events which can be raised by drift detectors when the beginning
    or end of a drift is detected.

    Parameters
    ----------
    p_drift_status : bool
        Determines whether a new drift starts (True) or an existing drift ends (False).
    p_id : int
        Drift ID. Default value = 0.
    p_tstamp : datetime
        Time stamp of drift detection. Default = None.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_raising_object : object
        Reference of the object raised. Default = None.
    **p_kwargs
        Further optional keyword arguments.

    Attributes
    ----------
    event_id : str
        Event id to be used when raising a drift event object. It is a string consisting of the 
        class name and one of the postfixes '(ON)', '(OFF)' depending on the drift status.
    """

    pass


