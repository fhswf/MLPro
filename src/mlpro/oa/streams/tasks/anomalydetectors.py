## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.tasks.boundarydetectors
## -- Module  : anomalydetectors.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SP       Creation
## -- 2023-                 SP       Release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-06-23)
This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.basics import Instance, List


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetector(OATask):
    """
    This is the base class for multivariate online anomaly detectors. It raises an event when an
    anomaly is detected.

    """

    C_NAME          = 'Anomaly Detector'
    C_TYPE          = 'Anomaly Detector'
    C_EVENT_ANOMALY = 'ANOMALY'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = None,
                 p_range_max = StreamTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)
        
        self.data_points = []
        self.anomaly_scores = None
        self.counter = 0
        self.anomaly_scores = []


    ## ------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, p_inst_del: list):
        pass



## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
# Custom class that inherits event management functionalities from MLPro's class EventManager
class AnomalyEvent (EventManager):

    C_NAME          = 'Event class'

    C_EVENT_OWN     = 'ANOMALY'

    def __init__(self, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging)
 

    def raise_anomaly(self, data_points, anomaly_scores, counter):
        eventobj = Event(p_raising_object=self, p_par1='Anomaly detected')
        self._raise_event(self.C_EVENT_OWN, eventobj)
        self.anomaly_characteristics(data_points, anomaly_scores, counter)


    def anomaly_characteristics(self, data_points, anomaly_scores, counter):
            
            count = 0
            for x in anomaly_scores:
                if x < 0:
                    count += 1
            
            frequency = len(data_points)/count



## ---------------------------------------------------------
## ---------------------------------------------------------
# Custom event handler class
class AnomalyEventHandler (Log):

    C_TYPE          = 'Event handler'
    C_NAME          = 'Anomaly-Event handler'

    def myhandler(self, p_event_id, p_event_object:Event):
        self.log(Log.C_LOG_TYPE_I, 'Received event id', p_event_id)
        self.log(Log.C_LOG_TYPE_I, 'Event data:', p_event_object.get_data())



## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class DriftEvent (AnomalyEvent):

    C_NAME          = 'Drift Anomaly'

    C_EVENT_OWN     = 'DRIFT'

    def __init__(self, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging)
 

    def do_something(self, data_points, anomaly_scores, counter):
        eventobj = Event(p_raising_object=self, p_par1='Drift detected')
        self._raise_event(self.C_EVENT_OWN, eventobj)

