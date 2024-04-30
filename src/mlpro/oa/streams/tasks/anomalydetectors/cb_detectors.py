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
        
        self._centroids = []
        self._clusterer = p_clusterer
        self._num_clusters = 0


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: list, center: float, centroids: list):
        pass




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterSizeChangeDetector(AnomalyDetectorCB):
    """
    This is the class for detecting change of size of clusters.

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


    def __init__(self, n_clusters=1):
        self.n_clusters = n_clusters
        self.cluster_centers = None
        self.cluster_radii = None
        self.prev_radius = None

    def fit(self, data):
        # Use k-means clustering to identify clusters
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(data)
        
        # Store cluster centers and compute radii
        self.cluster_centers = kmeans.cluster_centers_
        self.cluster_radii = self._compute_cluster_radii(data, kmeans.labels_)
        
        # Compute initial cluster radius
        self.prev_radius = np.max(self.cluster_radii)

    def _compute_cluster_radii(self, data, labels):
        radii = []
        for i in range(self.n_clusters):
            # Select data points belonging to the current cluster
            cluster_points = data[labels == i]
            # Compute distances from cluster center to all points in the cluster
            distances = np.linalg.norm(cluster_points - self.cluster_centers[i], axis=1)
            # Compute the radius of the cluster as the maximum distance
            cluster_radius = np.max(distances)
            radii.append(cluster_radius)
        return radii

    def detect_radius_change(self, new_data):
        # Update cluster centers and radii with new data
        self.cluster_centers = self._update_cluster_centers(new_data)
        self.cluster_radii = self._compute_cluster_radii(new_data, self._predict_clusters(new_data))
        
        # Compute current maximum cluster radius
        current_radius = np.max(self.cluster_radii)
        
        # Check for significant change in radius
        radius_change = current_radius - self.prev_radius
        self.prev_radius = current_radius  # Update previous radius
        
        return current_radius, radius_change

    def _predict_clusters(self, data):
        return KMeans(n_clusters=self.n_clusters, init=self.cluster_centers).fit_predict(data)

    def _update_cluster_centers(self, new_data):
        kmeans = KMeans(n_clusters=self.n_clusters, init=self.cluster_centers)
        kmeans.fit(new_data)
        return kmeans.cluster_centers_




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
        pass




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterWeightChangeDetector(AnomalyDetectorCB):
    """
    This is the class for detecting change in weight of clusters.

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

