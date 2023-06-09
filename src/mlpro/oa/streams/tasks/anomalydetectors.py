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
Ver. 0.0.0 (2023-06-08)
This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.basics import Instance, List
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest




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


    ## ------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, p_inst_del: list):
        pass


    ## ------------------------------------------------------------------------------------------------
    def _plot( p_figure:Figure=None,
               p_plot_settings : PlotSettings = None ):
        pass



## -------------------------------------------------------------------------
## -------------------------------------------------------------------------
# Custom class that inherits event management functionalities from MLPro's class EventManager
class AnomalyEvent (EventManager):

    C_NAME          = 'Event class'

    C_EVENT_OWN     = 'ANOMALY'

    def __init__(self, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging)
 

    def do_something(self):
        eventobj = Event(p_raising_object=self, p_par1='Anomaly detected')
        self._raise_event(self.C_EVENT_OWN, eventobj)


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

    C_NAME          = 'Drift class'

    C_EVENT_OWN     = 'DRIFT'

    def __init__(self, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging)
 

    def do_something(self):
        eventobj = Event(p_raising_object=self, p_par1='Drift detected')
        self._raise_event(self.C_EVENT_OWN, eventobj)

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class LOF(AnomalyDetector):

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
    ## ------------------------------------------------------------------------------------------------

    def _run(self, p_inst_new: list, p_inst_del: list):

        # Instance of the LOF algorithm
        lof = LocalOutlierFactor(self.num_neighbors)

        self.data_points.append(p_inst_new)

        # Perform anomaly detection
        self.anomaly_scores = lof.fit_predict(self.data_points)
            
        # Determine if data point is an anomaly based on its outlier score
        if self.anomaly_scores[-1] == -1:
            self.counter += 1
            handler_obj = AnomalyEventHandler()
            event_obj = AnomalyEvent()
            event_obj.register_event_handler(AnomalyEvent.C_EVENT_OWN, handler_obj.myhandler)
            event_obj.do_something()



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SVM(AnomalyDetector):

    C_NAME          = 'SVM Anomaly Detector'
    C_TYPE          = 'Anomaly Detector'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_kernel = 'rbf',
                 p_nu = 0.01,
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
        
        self.kernel = p_kernel
        self.nu = p_nu
    ## ------------------------------------------------------------------------------------------------

    def _run(self, p_inst_new: list, p_inst_del: list):

        # Instance of the LOF algorithm
        svm = OneClassSVM(kernel=self.kernel, nu=self.nu)

        self.data_points.append(p_inst_new)

        # Perform anomaly detection on the current data points
        svm.fit(self.data_points)
        self.anomaly_scores = svm.decision_function(self.data_points)
            
        # Determine if the data point is an anomaly based on its outlier score
        if self.anomaly_scores[-1] < 0:
            self.counter += 1
            print("Anomaly detected:", p_inst_new)



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IF(AnomalyDetector):

    C_NAME          = 'Isolation Forest Anomaly Detector'
    C_TYPE          = 'Anomaly Detector'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_estimators = 100,
                 p_contamination = 0.01,
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
        
        self.num_estimators = p_estimators
        self.contamination = p_contamination
    ## ------------------------------------------------------------------------------------------------

    def _run(self, p_inst_new: list, p_inst_del: list):

        # Instance of the LOF algorithm
        isolation_forest = IsolationForest(n_estimators=self.num_estimators,
                                           contamination=self.contamination)

        self.data_points.append(p_inst_new)

        # Perform anomaly detection on the current data points
        isolation_forest.fit(self.data_points)
        self.anomaly_scores = isolation_forest.decision_function(self.data_points)

        # Determine if the latest data point is an anomaly based on its outlier score
        if self.anomaly_scores[-1] < 0:
            counter += 1
            print("Anomaly detected:", p_inst_new)


