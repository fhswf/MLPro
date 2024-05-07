## -- ----------------------------------------------------------------------------------------------
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
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased import *
from mlpro.oa.streams.tasks.clusteranalyzers.basics import ClusterAnalyzer
import numpy as np



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
                 p_clusterer : ClusterAnalyzer = None,
                 p_name : str = None,
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
        
        self._clusterer = p_clusterer
        self._cluster_ids = []
        self._num_clusters = 0
        self._ref_centroids = {}
        self._centroids = {}
        self._ref_spacial_sizes = {}
        self._spacial_sizes = {}
        self._ref_velocities = {}
        self._velocities = {}
        self._ref_weights = {}
        self._weights = {}




## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, center: float, centroids: list):
        pass




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterGeometricSizeChangeDetector(AnomalyDetectorCB):
    """
    This is the class for detecting change of spatial size of clusters.

    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_clusterer : ClusterAnalyzer = None,
                 p_threshold_upper_limit : float = None,
                 p_threshold_lower_limit : float = None,
                 p_threshold_detection : float = None,
                 p_threshold_rate_of_change : float = None,
                 p_name:str = None,
                 p_range_max = StreamTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_clusterer = p_clusterer,
                         p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)
        
        self._thresh_ul = p_threshold_upper_limit
        self._thresh_ll = p_threshold_lower_limit
        self._thresh_det = p_threshold_detection
        self._thresh_roc = p_threshold_rate_of_change


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, center: float, centroids: list):

        inst = p_inst_new[-1].get_feature_data()
        feature_values = inst.get_values()

        cluster_id = 1

        if cluster_id not in self._cluster_ids:
            self._cluster_ids.append(cluster_id)

        if cluster_id not in self._ref_centroids.keys():
            self._ref_centroids[cluster_id] = self._clusterer.get_clusters()[cluster_id]
            self._centroids[cluster_id] = self._ref_centroids[cluster_id]
            center = self._centroids[cluster_id]
            self._ref_spacial_sizes[cluster_id] =  np.linalg.norm(feature_values - center)
            self._spacial_sizes[cluster_id] = self._ref_spacial_sizes[cluster_id]

        else:
            center = self._centroids[cluster_id]
            distance = np.linalg.norm(feature_values - center)
            if self._threshold != None:
                if (distance-self._ref_spacial_sizes[cluster_id]) > self._threshold:
                    self._ref_spacial_sizes[cluster_id] = distance
                    self._spacial_sizes[cluster_id] = self._ref_spacial_sizes[cluster_id]
                    event = ClusterEnlargement(p_instances=p_inst_new)
                else:
                    self._spacial_sizes[cluster_id] = distance
            else:
                if (distance-self._ref_spacial_sizes[cluster_id]) > self._ref_spacial_sizes[cluster_id]*0.05:
                    self._ref_spacial_sizes[cluster_id] = distance
                    self._spacial_sizes[cluster_id] = self._ref_spacial_sizes[cluster_id]
                    event = ClusterEnlargement(p_instances=p_inst_new)
                else:
                    self._spacial_sizes[cluster_id] = distance




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterVelocityChangeDetector(AnomalyDetectorCB):
    """
    This is the class for detecting change in velocity of clusters.

    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_clusterer : ClusterAnalyzer = None,
                 p_name:str = None,
                 p_range_max = StreamTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_clusterer = p_clusterer,
                         p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, center: float, centroids: list):
        # Fit the model with new data to update cluster centroids
        self.model.partial_fit()
        current_centroids = self.model.cluster_centers_

        # Calculate the velocity of each cluster
        velocities = np.linalg.norm(current_centroids - self.prev_centroids, axis=1)

        # Update previous centroids for the next iteration
        self.prev_centroids = current_centroids

        # Check for significant changes in cluster velocities
        mean_velocity = np.mean(velocities)
        max_velocity = np.max(velocities)
        velocity_threshold = 0.1  # Adjust this threshold as needed

        if max_velocity > velocity_threshold:
            return True, mean_velocity, max_velocity
        else:
            return False, mean_velocity, max_velocity




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterDensityChangeDetector(AnomalyDetectorCB):
    """
    This is the class for detecting change of density of clusters.

    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_clusterer : ClusterAnalyzer = None,
                 p_threshold_upper_limit : float = None,
                 p_threshold_lower_limit : float = None,
                 p_threshold_detection : float = None,
                 p_threshold_rate_of_change : float = None,
                 p_name:str = None,
                 p_range_max = StreamTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_clusterer = p_clusterer,
                         p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)
        
        self._thresh_ul = p_threshold_upper_limit
        self._thresh_ll = p_threshold_lower_limit
        self._thresh_det = p_threshold_detection
        self._thresh_roc = p_threshold_rate_of_change


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, center: float, centroids: list):

        pass



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterSizeChangeDetector(AnomalyDetectorCB):
    """
    This is the class for detecting change in weight of clusters.

    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_clusterer : ClusterAnalyzer = None,
                 p_scale : int = 1,
                 p_name:str = None,
                 p_range_max = StreamTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_clusterer = p_clusterer,
                         p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)
        
        self._scale = p_scale


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, center: float, centroids: list):
        pass




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterRelativeSizeChangeDetector(AnomalyDetectorCB):
    """
    This is the class for detecting change in weight of clusters.

    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_clusterer : ClusterAnalyzer = None,
                 p_scale : int = 1,
                 p_name:str = None,
                 p_range_max = StreamTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_clusterer = p_clusterer,
                         p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)
        
        self._scale = p_scale


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, center: float, centroids: list):
        pass




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NewClusterDetector(AnomalyDetectorCB):
    """
    This is the class for detecting new clusters.

    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_clusterer : ClusterAnalyzer = None,
                 p_name:str = None,
                 p_range_max = StreamTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_clusterer = p_clusterer,
                         p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)
        

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, center: float, centroids: list):
        
        clusters = self._clusterer.get_clusters()
        if len(clusters) > self._num_clusters:
            print((len(clusters)-self._num_clusters), "new clusters appeared.")
            event = NewClusterAppearance()
        self._num_clusters = len(clusters)




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterDisappearanceDetector(AnomalyDetectorCB):
    """
    This is the class for detecting the disappearences of clusters.

    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_clusterer : ClusterAnalyzer = None,
                 p_name:str = None,
                 p_range_max = StreamTask.C_RANGE_THREAD,
                 p_ada : bool = True,
                 p_duplicate_data : bool = False,
                 p_visualize : bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        super().__init__(p_clusterer = p_clusterer,
                         p_name = p_name,
                         p_range_max = p_range_max,
                         p_ada = p_ada,
                         p_duplicate_data = p_duplicate_data,
                         p_visualize = p_visualize,
                         p_logging = p_logging,
                         **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, center: float, centroids: list):
        
        clusters = self._clusterer.get_clusters()
        if len(clusters) < self._num_clusters:
            print((self._num_clusters-len(clusters)), "clusters disappeared")
            event = ClusterDisappearence()
        self._num_clusters = len(clusters)

