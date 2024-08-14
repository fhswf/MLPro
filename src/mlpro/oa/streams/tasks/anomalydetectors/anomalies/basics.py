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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.0 (2024-05-22)

This module provides a template class for anomalies to be used in anomaly detection algorithms.
"""

from mlpro.bf.various import Id
from mlpro.bf.plot import Plottable, PlotSettings
from mlpro.bf.events import Event
from mlpro.bf.streams import Instance
from mlpro.bf.math.properties import PropertyDefinitions, Properties





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Anomaly (Id, Event, Plottable):
    """
    This is the base class for anomaly events which can be raised by the anomaly detectors when an
    anomaly is detected.

    Parameters
    ----------
    p_id : int
        Anomaly ID. Default value = 0.
    p_instances : Instance
        List of instances. Default value = None.
    p_ano_scores : list
        List of anomaly scores of instances. Default = None.
    p_det_time : str
        Time of occurance of anomaly. Default = None.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_raising_object : object
        Reference of the object raised. Default = None.
    **p_kwargs
        Further optional keyword arguments.
    """

    C_TYPE                  = 'Anomaly'

    C_PLOT_ACTIVE           = True
    C_PLOT_STANDALONE       = False
    C_PLOT_VALID_VIEWS      = [ PlotSettings.C_VIEW_2D, 
                                PlotSettings.C_VIEW_3D, 
                                PlotSettings.C_VIEW_ND ]
    C_PLOT_DEFAULT_VIEW     = PlotSettings.C_VIEW_ND

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_id : int = 0,
                 p_instances: list[Instance] = None,
                 p_ano_scores : list = None,
                 p_det_time : str = None,
                 p_visualize : bool = False,
                 p_raising_object : object = None,
                 **p_kwargs):
        
        Id.__init__( self, p_id = p_id )
        Event.__init__( self, p_raising_object=p_raising_object,
                        p_tstamp=p_det_time, **p_kwargs)
        Plottable.__init__( self, p_visualize = p_visualize )

        self._instances : list[Instance] = p_instances
        self._ano_scores = p_ano_scores


## -------------------------------------------------------------------------------------------------
    def get_instances(self) -> list[Instance]:
        """
        Method that returns the instances associated with the anomaly.
        
        Returns
        -------
        list[Instance]
            The list of instances.
        """
        return self._instances
    

## -------------------------------------------------------------------------------------------------
    def get_detection_time(self) -> float:
        """
        Method that returns the time at which the anomaly/anomalies were detection.
        
        Returns
        -------
        float
            The time of detection.
        """
        return self._instances
    

## -------------------------------------------------------------------------------------------------
    def get_ano_scores(self):
        """
        Method that returns the anomaly scores associated with the instances of the anomaly.
        
        Returns
        -------
        list
            The list of anomaly scores.
        """
        return self._ano_scores
    

