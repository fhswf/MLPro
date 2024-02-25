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
## -- 2024-02-25  1.0.2     SP       Visualisation update
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2023-02-25)
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
    C_EVENT_ANOMALY = 'ANOMALY'
    C_ANOMALY_TYPES = ['Point Anomaly', 'Group Anomaly', 'Contextual Anomaly']


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
        self.instance_id = 0
        self.anomaly_counter = 0
        self.anomaly_scores = []
        self.anomalies = [ [], [], [], []]
        self.anomaly_type = 'Point Anomaly'
        self.time_of_occurance = 0
        self.plot_update_counter = 0
        #self.anomalies = [['instance'],['feature'],['type'],['time']]


## ------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, p_inst_del: list):
        pass


## ------------------------------------------------------------------------------------------------
    def _raise_anomaly(self, p_event_id:str, p_event_object:Event, p_instance : Instance):
        time_stamp = p_instance.get_time_stamp()
        self._raise_event(p_event_id, p_event_object)


## ------------------------------------------------------------------------------------------------
    def def_anomalies(self):
        self.anomalies[0].append(self.instance_id)
        self.anomalies[1].append(2)
        self.anomalies[2].append(self.anomaly_type)
        self.anomalies[3].append(self.time_of_occurance)

        if len(self.anomalies[0]) > 100:
             for i in range(len(self.anomalies)):
                 self.anomalies[i].pop(0)

        print(self.anomalies)


    """## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure: Figure, p_settings: PlotSettings):
        self._plot_line1 = None
        self._plot_line1_t1 : Text = None"""

                 
## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings: PlotSettings, **p_kwargs):
        super()._update_plot_nd(p_settings, **p_kwargs)

        if self.plot_update_counter < len(self.anomalies[0]):

            ylim  = p_settings.axes.get_ylim()
            label = str(self.anomalies[2][-1]) + ' at ' + str(self.anomalies[3][-1])
            self._plot_line1 = p_settings.axes.plot( [self.anomalies[0][-1], self.anomalies[0][-1]], ylim,
                                                    color='r', linestyle='dashed', lw=1, label=label)[0]
            self._plot_line1_t1 = p_settings.axes.text(self.anomalies[0][-1], 0, label, color='r' )

            self.plot_update_counter = self.plot_update_counter + 1


"""## ------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure: Figure = None, p_plot_settings : PlotSettings= None, **p_kwargs):
        super().init_plot(p_figure=p_figure, p_plot_settings=p_plot_settings, **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure:Figure, p_settings:PlotSettings):
        if p_settings.axes is None:
            p_settings.axes = p_figure.add_subplot( p_settings.pos_y, p_settings.pos_x, p_settings.id )"""





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class LocalOutlierFactor(AnomalyDetector):
    C_NAME          = 'LOF Anomaly Detector'
    C_TYPE          = 'Anomaly Detector'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_neighbours = 10,
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
        
        self.num_neighbours = p_neighbours
        self.lof = LOF(self.num_neighbours)


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, p_inst_del: list):
        det_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.adapt(p_inst_new, p_inst_del)

        # Determine if data point is an anomaly based on its outlier score
        if -1 in self.anomaly_scores:
            self.anomaly_counter = self.anomaly_counter + 1
            print("Anomaly detected", self.anomaly_counter)
            self.def_anomalies()
            print(self.instance_id)

        
        
        """event_obj = AnomalyEvent(p_raising_object=self, p_det_time=det_time,
                                     p_instance=str(self.data_points[-1]))
            handler = self.event_handler
            self.register_event_handler(event_obj.C_NAME, handler)
            self._raise_event(event_obj.C_NAME, event_obj)"""


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new):
        for inst in p_inst_new:
            if isinstance(inst, Instance):
                feature_data = inst.get_feature_data()
            else:
                feature_data = inst

        values = feature_data.get_values()
        self.instance_id = inst.get_id()
        
        self.anomaly_scores = []
        if len(self.data_points) == 0:
            for i in range(len(values)):
                self.data_points.append([])

        i=0
        for value in values:
            self.data_points[i].append(value)
            i=i+1

        if len(self.data_points[0]) > 100:
            for i in range(len(values)):
                self.data_points[i].pop(0)

        if len(self.data_points[0]) >= 3:
            for i in range(len(values)):
                scores = self.lof.fit_predict(np.array(self.data_points[i]).reshape(-1, 1))
                self.anomaly_scores.append(scores[-1])
        print(self.anomaly_scores)


## -------------------------------------------------------------------------------------------------
    def event_handler(self, p_event_id, p_event_object:Event):
        self.log(Log.C_LOG_TYPE_I, 'Received event id', p_event_id)





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
        self.anomaly_counter = 0
        self.anomaly_scores = []
        self.threshold = p_threshold
        self.centroid_thre = p_centroid_threshold
        self.centroids = []


# -------------------------------------------------------------------------
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
    




## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class AnomalyEvent (Event):

    C_TYPE     = 'Event'
    C_NAME     = 'Anomaly'

# -------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time : str, p_instance: str, **p_kwargs):
        pass





## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class PointAnomaly (AnomalyEvent):

    C_NAME      = 'Point Anomaly'

# -------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time : str, p_instance : str, p_deviation : float, **p_kwargs):
        pass





## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class GroupAnomaly (AnomalyEvent):

    C_NAME      = 'Group Anomaly'

# -------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time : str, p_instances : list, p_mean : float, p_mean_deviation : float, **p_kwargs):
        pass





## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class ContextualAnomaly (AnomalyEvent):

    C_NAME      = 'Contextual Anomaly'

# -------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time :str, p_instance: str,  **p_kwargs):
        pass





## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class DriftEvent (AnomalyEvent):

    C_NAME      = 'Drift'

# -------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time : str, p_magnitude : float, p_rate : float, **p_kwargs):
        pass





## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
class DriftEventCB (DriftEvent):

    C_NAME      = 'Cluster based Drift'

# -------------------------------------------------------------------------
    def __init__(self, p_raising_object, p_det_time : str, **p_kwargs):
        pass




