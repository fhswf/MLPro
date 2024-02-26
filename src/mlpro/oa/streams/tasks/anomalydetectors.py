## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.tasks.anomalydetectors
## -- Module  : anomalydetectors.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SP       Creation
## -- 2023-09-12  1.0.0     SP       Release
## -- 2023-11-21  1.0.1     SP       Time Stamp update
## -- 2024-02-25  1.1.0     SP       Visualisation update
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2023-02-25)
This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.basics import Instance, List
import numpy as np
from matplotlib.text import Text
from sklearn.neighbors import LocalOutlierFactor as LOF
from datetime import datetime




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetector(OATask):
    """
    This is the base class for multivariate online anomaly detectors. It raises an event when an
    anomaly is detected.

    """

    C_NAME          = 'Anomaly Detector'
    C_TYPE          = 'Anomaly Detector'

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
        self.data_size = 100
        self.inst_id = 0
        self.inst_value = 0
        self.ano_counter = 0
        self.ano_scores = []
        self.anomalies = {'inst_id':[], 'inst_value':[], 'ano_score':[], 'ano_type':[],
                          'time_of_occ':[]}
        self.ano_type = 'Anomaly'
        self.time_of_occ = 0
        self.plot_update_counter = 0
        self.consec_count = 1
        self.group_anomalies = []


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, p_inst_del: list):
        pass


## -------------------------------------------------------------------------------------------------
    def raise_anomaly_event(self, p_instance : Instance):
        p_instance = p_instance
        self.time_of_occ = p_instance.get_tstamp()
        self.inst_id = p_instance.get_id()
        self.ano_type = self.get_anomaly_type(p_instance)

        self.ano_counter += 1
        self.anomalies['inst_id'].append(self.inst_id)
        self.anomalies['inst_value'].append(self.inst_value)
        self.anomalies['ano_score'].append(self.ano_scores)
        self.anomalies['ano_type'].append(self.ano_type)
        self.anomalies['time_of_occ'].append(self.time_of_occ)

        if len(self.anomalies['inst_id']) > 100:
             for x in self.anomalies:
                 self.anomalies[x].pop(0)

        if self.ano_type == 'Point Anomaly':
            event = PointAnomaly(p_raising_object=self, p_det_time=str(self.time_of_occ),
                                 p_instance=p_instance)
        elif self.ano_type == 'Group Anomaly':
            event = GroupAnomaly(self, p_det_time=self.time_of_occ,
                                 p_instances=self.group_anomalies)

        self._raise_event(event.C_NAME, event)


## -------------------------------------------------------------------------------------------------
    def get_anomaly_type(self, p_instance):
        self.ano_type = 'Point Anomaly'

        if len(self.anomalies['inst_id']) > 1:
            if int(self.anomalies['inst_id'][-1]) - 1 == int(self.anomalies['inst_id'][-2]):
                self.consec_count +=1
                self.group_anomalies.append(p_instance)
                if self.consec_count > 2:
                    self.ano_type = 'Group Anomaly'
                    return self.ano_type
                else:
                    self.ano_type = 'Point Anomaly'
                    return self.ano_type
            else:
                self.consec_count = 1
                self.ano_type = 'Point Anomaly'
                self.group_anomalies = []
                self.group_anomalies.append(p_instance)
                return self.ano_type
        else:
            self.ano_type = 'Point Anomaly'
            self.group_anomalies.append(p_instance)
            return self.ano_type

                 
## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings: PlotSettings, **p_kwargs):
        super()._update_plot_nd(p_settings, **p_kwargs)

        if self.plot_update_counter < len(self.anomalies['inst_id']):
            ylim  = p_settings.axes.get_ylim()
            label = str(self.anomalies['ano_type'][-1][0])
            self._plot_line1 = p_settings.axes.plot([self.anomalies['inst_id'][-1], self.anomalies['inst_id'][-1]],
                                                    ylim, color='r', linestyle='dashed', lw=1, label=label)[0]
            self._plot_line1_t1 = p_settings.axes.text(self.anomalies['inst_id'][-1], 0, label, color='r' )
            self.plot_update_counter = self.plot_update_counter + 1





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
            event_obj = AnomalyEvent(p_raising_object=self, p_kwargs=self.data_points[-1]) 
            handler = self.event_handler
            self.register_event_handler(event_obj.C_NAME, handler)
            self._raise_event(event_obj.C_NAME, event_obj)
    




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyEvent (Event):
    """
    This is the base class for anomaly events which can be raised by the anomaly detectors when an
    anomaly is detected.

    """

    C_TYPE     = 'Event'
    C_NAME     = 'Anomaly'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time : str, **p_kwargs):
        super().__init__(p_raising_object=p_raising_object,
                         p_tstamp=p_det_time, **p_kwargs)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PointAnomaly (AnomalyEvent):
    """
    Event class for anomaly events when point anomalies are detected.
    
    """

    C_NAME      = 'Point Anomaly'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time : str=None, p_instance : str=None,
                 p_deviation : float=None, **p_kwargs):
        super().__init__(p_raising_object=p_raising_object, p_det_time=p_det_time, **p_kwargs)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GroupAnomaly (AnomalyEvent):
    """
    Event class for anomaly events when group anomalies are detected.
    
    """

    C_NAME      = 'Group Anomaly'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time : str, p_instances : list=None,
                 p_mean : float=None, p_mean_deviation : float=None, **p_kwargs):
        super().__init__(p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ContextualAnomaly (AnomalyEvent):
    """
    Event class for anomaly events when contextual anomalies are detected
    
    """

    C_NAME      = 'Contextual Anomaly'

# -------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time :str, p_instances: str,  **p_kwargs):
        super().__init__(p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftEvent (AnomalyEvent):
    """
    Event class to be raised when drift is detected.
    
    """

    C_NAME      = 'Drift'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time : str, p_magnitude : float, p_rate : float, **p_kwargs):
        super().__init__(p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DriftEventCB (DriftEvent):
    """
    Event class to be raised when cluster drift is detected.
    
    """

    C_NAME      = 'Cluster based Drift'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time : str, **p_kwargs):
        super().__init__(p_raising_object=p_raising_object,
                         p_det_time=p_det_time, **p_kwargs)

