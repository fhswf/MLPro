## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.anomalies
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -- 2024-04-11  1.3.0     DA       Class Anomaly: extensions on methods update_plot_*
## -- 2024-05-07  1.3.1     SK       Bug fix related to p_instances
## -- 2024-05-09  1.3.2     DA       Bugfix in method Anomaly._update_plot()
## -- 2024-05-22  1.4.0     SK       Refactoring
## -- 2025-02-12  1.4.1     DA       Code reduction
## -- 2025-02-18  2.0.0     DA       Class Anomaly:
## --                                - refactoring and simplification
## --                                - new attribute event_id
## --                                - new parent Renormalizable
## -- 2025-05-12  2.1.0     DS       Design extention : new method _get_status()
## -- 2025-05-20  2.1.1     DA\DS    Design extension 
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.1.1 (2025-05-20)

This module provides a template class for anomalies to be used in anomaly detection algorithms.
"""

from datetime import datetime

from mlpro.bf.various import Id
from mlpro.bf.plot import Plottable, PlotSettings
from mlpro.bf.events import Event
from mlpro.bf.math.normalizers import Renormalizable




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Anomaly (Id, Event, Plottable, Renormalizable):
    """
    This is the base class for anomaly events which can be raised by the anomaly detectors when an
    anomaly is detected.

    Parameters
    ----------
    p_id : int
        Anomaly ID. Default value = 0.
    p_tstamp : datetime
        Time of occurance of anomaly. Default = None.
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
                 p_status : bool = True,
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

        self._status: bool = p_status
        
        if p_status:
            self._event_id = type(self).__name__ + '(ON)'
        else:
            self._event_id = type(self).__name__ + '(OFF)'


## -------------------------------------------------------------------------------------------------
    def _get_status(self) -> bool:
        return self._status


## -------------------------------------------------------------------------------------------------
    def _get_event_id(self) -> str:
        return self._event_id


## -------------------------------------------------------------------------------------------------
    status       = property( fget = _get_status )
    event_id     = property( fget = _get_event_id ) 