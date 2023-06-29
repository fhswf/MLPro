## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers.sklearn
## -- Module  : anomalydetectors.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-23  1.0.0     SY       Creation
## -- 2023-06-23  1.0.0     SY       First version release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-06-23)

This module provides wrapper functionalities to incorporate anomaly detector algorithms of the 
Scikit-learn ecosystem. This module includes three algorithms from Scikit-learn that are embedded to MLPro, such as:

1) Local Outlier Factor (LOF)
2) One Class SVM
3) Isolation Forest (IF)

Learn more:
https://scikit-learn.org

"""

from mlpro.wrappers.sklearn.basics import *
from mlpro.oa.streams.basics import Instance, List
from mlpro.oa.streams.tasks.anomalydetectors import *
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class LOF(AnomalyDetector):

    C_NAME          = 'LOF Anomaly Detector'
    C_TYPE          = 'Anomaly Detector'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_neighbours = 10,
                 p_sizeof_instance_list = 100,
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
        self.sizeof_instance_list = p_sizeof_instance_list
    ## ------------------------------------------------------------------------------------------------

    def _run(self, p_inst_new: list, p_inst_del: list):

        self.data_points.append(p_inst_new)

        if len(self.data_points) > self.sizeof_instance_list:
            self.data_points.pop(0)

        if len(self.data_points) >= 20:

            # Instance of the LOF algorithm
            lof = LocalOutlierFactor(self.num_neighbors)

            # Perform anomaly detection
            self.anomaly_scores = lof.fit_predict(self.data_points)
                
            # Determine if data point is an anomaly based on its outlier score
            if self.anomaly_scores[-1] == -1:
                event_obj = AnomalyEvent(p_raising_object=self, p_kwargs=self.data_points[-1])                )
                handler = self.myhandler
                self.register_event_handler(event_obj.C_NAME, handler)
                self._raise_event(event_obj.C_NAME, event_obj)



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SVM(AnomalyDetector):

    C_NAME          = 'SVM Anomaly Detector'
    C_TYPE          = 'Anomaly Detector'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_kernel = 'rbf',
                 p_nu = 0.01,
                 p_sizeof_instance_list = 100,
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
        self.sizeof_instance_list = p_sizeof_instance_list
    ## ------------------------------------------------------------------------------------------------

    def _run(self, p_inst_new: list, p_inst_del: list):

        self.data_points.append(p_inst_new)

        if len(self.data_points) > self.sizeof_instance_list:
            self.data_points.pop(0)

        if len(self.data_points) >= 20:

            # Instance of the LOF algorithm
            svm = OneClassSVM(kernel=self.kernel, nu=self.nu)

            # Perform anomaly detection on the current data points
            svm.fit(self.data_points)
            self.anomaly_scores = svm.decision_function(self.data_points)
                
            # Determine if the data point is an anomaly based on its outlier score
            if self.anomaly_scores[-1] < 0:
                self.counter += 1
                event_obj = AnomalyEvent(p_raising_object=self, p_kwargs=self.data_points[-1]) 
                handler = self.myhandler
                self.register_event_handler(event_obj.C_NAME, handler)
                self._raise_event(event_obj.C_NAME, event_obj)



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IF(AnomalyDetector):

    C_NAME          = 'Isolation Forest Anomaly Detector'
    C_TYPE          = 'Anomaly Detector'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_estimators = 100,
                 p_contamination = 0.01,
                 p_sizeof_instance_list = 100,
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
        self.sizeof_instance_list = p_sizeof_instance_list
    ## ------------------------------------------------------------------------------------------------

    def _run(self, p_inst_new: list, p_inst_del: list):

        self.data_points.append(p_inst_new)

        if len(self.data_points) > self.sizeof_instance_list:
            self.data_points.pop(0)

        if len(self.data_points) >= 20:

            # Instance of the LOF algorithm
            isolation_forest = IsolationForest(n_estimators=self.num_estimators,
                                                contamination=self.contamination)

            # Perform anomaly detection on the current data points
            isolation_forest.fit(self.data_points)
            self.anomaly_scores = isolation_forest.decision_function(self.data_points)

            # Determine if the latest data point is an anomaly based on its outlier score
            if self.anomaly_scores[-1] < 0:
                self.counter += 1
                event_obj = AnomalyEvent(p_raising_object=self, p_kwargs=self.data_points[-1]) 
                handler = self.myhandler
                self.register_event_handler(event_obj.C_NAME, handler)
                self._raise_event(event_obj.C_NAME, event_obj)


