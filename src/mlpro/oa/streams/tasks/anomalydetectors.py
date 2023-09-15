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
import numpy as np


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



## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class AnomalyDetectorCB(AnomalyDetector):

    C_TYPE = 'Cluster based Anomaly Detector'


    ## ------------------------------------------------------------------------
    def __init__(self,
                 p_threshold = 5.0,
                 p_centroid_threshold = 1.0,
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
        self.threshold = p_threshold
        self.centroid_thre = p_centroid_threshold
        self.centroids = []



    ## -------------------------------------------------------------------------
    def _run(self, p_inst_new: list, center: float, centroids: list):

        anomaly = None
        self.centroids.append(centroids)
        
        distance = np.linalg.norm(p_inst_new - center)
        if distance > self.threshold:
            anomaly = p_inst_new

        if len(centroids) > 10:
            self.centroids.pop(0)
        
        if len(self.centroids[-2]) != len(self.centroids[-1]):
            anomaly = p_inst_new

        differences = [abs(a - b) for a, b in zip(self.centroids[0], self.centroids[-1])]
        if any(difference >= self.centroid_thre for difference in differences):
            anomlay = p_inst_new

        if anomaly != None:
            self.counter += 1
            event_obj = AnomalyEvent(p_raising_object=self, p_kwargs=self.data_points[-1]) 
            handler = self.event_handler
            self.register_event_handler(event_obj.C_NAME, handler)
            self._raise_event(event_obj.C_NAME, event_obj)
    


## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class AnomalyEvent (Event):

    C_TYPE     = 'Event'

    C_NAME     = 'Anomaly'

    def __init__(self, p_raising_object, p_det_time : str, p_instance: str, **p_kwargs):
        pass



## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class PointAnomaly (AnomalyEvent):

    C_NAME      = 'Point Anomaly'

    def __init__(self, p_raising_object, p_det_time : str, p_instance : str, p_deviation : float, **p_kwargs):
        pass


## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class GroupAnomaly (AnomalyEvent):

    C_NAME      = 'Group Anomaly'

    def __init__(self, p_raising_object, p_det_time : str, p_instances : list, p_mean : float, p_mean_deviation : float, **p_kwargs):
        pass


## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class ContextualAnomaly (AnomalyEvent):

    C_NAME      = 'Contextual Anomaly'

    def __init__(self, p_raising_object, p_det_time :str, p_instance: str,  **p_kwargs):
        pass


## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class DriftEvent (AnomalyEvent):

    C_NAME      = 'Drift'

    def __init__(self, p_raising_object, p_det_time : str, p_magnitude : float, p_rate : float, **p_kwargs):
        pass


## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class DriftEventCB (DriftEvent):

    C_NAME      = 'Cluster based Drift'

    def __init__(self, p_raising_object, p_det_time : str, **p_kwargs):
        pass

