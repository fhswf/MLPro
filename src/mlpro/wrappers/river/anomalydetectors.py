## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers.river
## -- Module  : anomalydetectors.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- 2023-06-23  1.0.0     SP       Creation
## -- 2023-06-23  1.0.0     SP       First version release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-05-12)

This module provides wrapper functionalities to incorporate anomaly detector algorithms of the 
River ecosystem. This module includes two algorithms from River that are embedded to MLPro, such as:

1) Half Space Tree (HST)
1) One Class SVM

Learn more:
https://www.riverml.xyz/

"""

from mlpro.wrappers.river.basics import *
from mlpro.oa.streams.basics import Instance, List
from mlpro.oa.streams.tasks.anomalydetectors import *
from river import anomaly

class HST(AnomalyDetector):

    C_NAME          = 'HST Anomaly Detector'
    C_TYPE          = 'Anomaly Detector'

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_no_trees = 5,
                 p_height = 3,
                 p_window_size = 3,
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
        
        self.no_trees = p_no_trees
        self.height = p_height
        self.window_size = p_window_size
        self.sizeof_instance_list = p_sizeof_instance_list
    ## ------------------------------------------------------------------------------------------------

    def _run(self, p_inst_new: list, p_inst_del: list):

        self.data_points.append(p_inst_new)

        if len(self.data_points) > self.sizeof_instance_list:
            self.data_points.pop(0)

        if len(self.data_points) >= 20:

            # Instance of the LOF algorithm
            hst = anomaly.HST(n_trees = self.no_trees, height = self.height, window_size = self.window_size, seed = 42)

            # Perform anomaly detection on the current data points
            hst.learn_one(self.data_points)
            self.anomaly_scores = hst.score_one(self.data_points[-1])
                
            # Determine if the data point is an anomaly based on its outlier score
            if self.anomaly_scores < 0:
                self.counter += 1
                event_obj = AnomalyEvent(p_raising_object=self, p_kwargs=self.data_points[-1]) 
                handler = self.event_handler
                self.register_event_handler(event_obj.C_NAME, handler)
                self._raise_event(event_obj.C_NAME, event_obj)



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SVM(AnomalyDetector):

    C_NAME          = 'OneClassSVM Anomaly Detector'
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
            svm = anomaly.SVM(p_kernel = self.kernel, p_nu = self.nu)

            # Perform anomaly detection on the current data points
            svm.learn_one(self.data_points)
            self.anomaly_scores = svm.score_one(self.data_points[-1])
                
            # Determine if the data point is an anomaly based on its outlier score
            if self.anomaly_scores < 0:
                self.counter += 1
                event_obj = AnomalyEvent(p_raising_object=self, p_kwargs=self.data_points[-1]) 
                handler = self.event_handler
                self.register_event_handler(event_obj.C_NAME, handler)
                self._raise_event(event_obj.C_NAME, event_obj)



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------