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
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.3.0 (2025-02-19)

This module provides a template class for types of data drift to be used in drift detection algorithms.
"""

from datetime import datetime

from mlpro.bf.various import Id
from mlpro.bf.plot import Plottable
from mlpro.bf.events import Event
from mlpro.bf.math.normalizers import Renormalizable



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Drift (Id, Event, Plottable, Renormalizable):
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

    C_PLOT_STANDALONE = False

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_drift_status : bool,
                  p_id : int = 0,
                  p_tstamp : datetime = None,
                  p_visualize : bool = False,
                  p_raising_object : object = None,
                  **p_kwargs):
        
        Id.__init__( self, p_id = p_id )

        Event.__init__( self, 
                        p_raising_object=p_raising_object,
                        p_tstamp=p_tstamp, 
                        **p_kwargs )
        
        Plottable.__init__( self, p_visualize = p_visualize )

        self._drift_status : bool = p_drift_status

        if p_drift_status:
            self._event_id = type(self).__name__ + '(ON)'
        else:
            self._event_id = type(self).__name__ + '(OFF)'


## -------------------------------------------------------------------------------------------------
    def _get_drift_status(self) -> bool:
        return self._drift_status
    

## -------------------------------------------------------------------------------------------------
    def _get_event_id(self) -> str:
        return self._event_id


## -------------------------------------------------------------------------------------------------
    drift_status = property( fget = _get_drift_status )
    event_id     = property( fget = _get_event_id ) 
