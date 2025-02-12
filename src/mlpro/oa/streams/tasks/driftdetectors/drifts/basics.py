## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.drifts
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-02-12  1.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-02-12)

This module provides a template class for types of data drift to be used in drift detection algorithms.
"""

from datetime import datetime

from mlpro.bf.various import Id
from mlpro.bf.plot import Plottable
from mlpro.bf.events import Event





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Drift (Id, Event, Plottable):
    """
    This is the base class for drift events which can be raised by the drift detectors when a
    drift is detected.

    Parameters
    ----------
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
    """

    C_TYPE                  = 'Drift'
    C_PLOT_STANDALONE       = False

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_id : int = 0,
                 p_tstamp : datetime = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 **p_kwargs):
        
        Id.__init__( self, p_id = p_id )
        Event.__init__( self, p_raising_object=p_raising_object,
                        p_tstamp=p_tstamp, **p_kwargs)
        Plottable.__init__( self, p_visualize = p_visualize )
