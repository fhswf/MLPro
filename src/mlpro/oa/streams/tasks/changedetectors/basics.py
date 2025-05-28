## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.changedetectors
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-05-28  0.1.0     DA/DS    Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-05-28)

This module provides templates for change detection to be used in the context of online adaptivity.
"""



from datetime import datetime

from mlpro.bf.various import Id
from mlpro.bf.plot import Plottable, PlotSettings
from mlpro.bf.events import Event
from mlpro.bf.math.normalizers import Renormalizable

from mlpro.oa.streams import OAStreamTask




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Change (Id, Event, Plottable, Renormalizable):
    """
    This is the base class for change events which can be raised by the change detectors when an
    change is detected.

    Parameters
    ----------
    p_id : int
        Change ID. Default value = 0.
    p_tstamp : datetime
        Time of occurance of change. Default = None.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_raising_object : object
        Reference of the object raised. Default = None.
    **p_kwargs
        Further optional keyword arguments.
    """

    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False
    C_PLOT_VALID_VIEWS      = [ PlotSettings.C_VIEW_2D, 
                                PlotSettings.C_VIEW_3D, 
                                PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW     = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def __init__(self,
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

        self.event_id   = type(self).__name__





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ChangeDetector (OAStreamTask):
    """
    Base class for online change detectors.

    Parameters
    ----------
    p_name : str
        Optional name of the task. Default is None.
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_ada : bool
        Boolean switch for adaptivitiy. Default = True.
    p_duplicate_data : bool
        If True, instances will be duplicated before processing. Default = False.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_anomaly_buffer_size : int = 100
        Size of the internal anomaly buffer self.anomalies. Default = 100.
    p_kwargs : dict
        Further optional named parameters.
    """

    C_TYPE              = 'Change Detector'

    C_PLOT_ACTIVE       = True
    C_PLOT_STANDALONE   = False

