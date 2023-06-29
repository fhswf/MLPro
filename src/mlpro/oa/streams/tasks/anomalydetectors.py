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
        self.counter = 0
        self.anomaly_scores = []


    ## ------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, p_inst_del: list):
        pass

    ## ---------------------------------------------------------------------------------
    def myhandler(self, p_event_id, p_event_object:Event):
        self.log(Log.C_LOG_TYPE_I, 'Received event id', p_event_id)
        self.log(Log.C_LOG_TYPE_I, 'Event data:', p_event_object.get_data())



## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class AnomalyDetectorCB(AnomalyDetector):

    C_TYPE = 'Cluster based Anomaly Detector'


    ## ------------------------------------------------------------------------
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
        self.counter = 0
        self.anomaly_scores = []


    ## -------------------------------------------------------------------------
    def _run(self, p_inst_new: list, p_inst_del: list):
        pass


    ## --------------------------------------------------------------------------
    def hdl_cluster_updates(p_event_id: str, p_event_object: Event):
        pass


## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class AnomalyEvent (Event):

    C_TYPE     = 'Event'

    C_NAME     = 'Anomaly'

    def __init__(self, p_raising_object, **p_kwargs):
        pass


## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class PointAnomaly (AnomalyEvent):

    C_NAME      = 'Point Anomaly'

    def __init__(self, p_raising_object, p_deviation, **p_kwargs):
        pass


## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class GroupAnomaly (AnomalyEvent):

    C_NAME      = 'Group Anomaly'

    def __init__(self, p_raising_object, p_mean, p_mean_deviation, **p_kwargs):
        pass


## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class ContextualAnomaly (AnomalyEvent):

    C_NAME      = 'Contextual Anomaly'

    def __init__(self, p_raising_object, **p_kwargs):
        pass


## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class DriftEvent (AnomalyEvent):

    C_NAME      = 'Drift'

    def __init__(self, p_raising_object, p_deviation, **p_kwargs):
        pass


## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class DriftEventCB (DriftEvent):

    C_NAME      = 'Cluster based Drift'

    def __init__(self, p_raising_object, p_deviation, **p_kwargs):
        pass

