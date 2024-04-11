## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors
## -- Module  : cb_detectors.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2024-04-10)

This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.tasks.anomalydetectors.basics import AnomalyDetector
from mlpro.oa.streams.tasks.anomalydetectors.anomalies import Anomaly



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorCB(AnomalyDetector):
    """
    This is the base class for cluster-based online anomaly detectors. It raises an event when an
    anomaly is detected in a cluster dataset.

    """

    C_TYPE = 'Cluster based Anomaly Detector'


## -------------------------------------------------------------------------------------------------
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
        self.anomaly_counter = 0
        self.anomaly_scores = []
        self.threshold = p_threshold
        self.centroid_thre = p_centroid_threshold
        self.centroids = []


## -------------------------------------------------------------------------------------------------
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
            anomaly = p_inst_new

        if anomaly != None:
            self.anomaly_counter += 1
            event_obj = Anomaly(p_raising_object=self, p_kwargs=self.data_points[-1]) 
            handler = self.event_handler
            self.register_event_handler(event_obj.C_NAME, handler)
            self._raise_event(event_obj.C_NAME, event_obj)
    