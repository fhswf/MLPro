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
## -- 2024-05-22  1.2.1     SK       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2024-04-10)

This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.tasks.anomalydetectors.basics import AnomalyDetector
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased import *
from mlpro.oa.streams.tasks.clusteranalyzers.basics import ClusterAnalyzer
from mlpro.bf.streams import Instance, InstDict
import numpy as np
from scipy.spatial.distance import cdist
from mlpro.bf.math.properties import *



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorCB(AnomalyDetector, Properties):
    """
    This is the base class for cluster-based online anomaly detectors. It raises an event when an
    anomaly is detected in a cluster dataset.

    """

    C_TYPE = 'Cluster based Anomaly Detector'

    # List of cluster properties necessary for the algorithm
    C_REQ_CLUSTER_PROPERTIES : PropertyDefinitions = []


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
        self._rel_weights = {}
        self._ref_rel_weights = {}




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterGeometricSizeChangeDetector(AnomalyDetectorCB):
    """
    This is the class for detecting change of spatial size of clusters.

    """
    
    C_PROPERTIY_DEFINITIONS : PropertyDefinitions = [ ['size', 2, Property]]

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
        
        for x in self.C_PROPERTIY_DEFINITIONS:
            if x not in self.C_REQ_CLUSTER_PROPERTIES:
                self.C_REQ_CLUSTER_PROPERTIES.append(x)

        unknown_prop = self._clusterer.align_cluster_properties(p_properties=self.C_REQ_CLUSTER_PROPERTIES)

        if len(unknown_prop) >0:
            raise RuntimeError("The following cluster properties need to be provided by the clusterer: ", unknown_prop)
        
        self._thresh_ul = p_threshold_upper_limit
        self._thresh_ll = p_threshold_lower_limit
        self._thresh_det = p_threshold_detection
        self._thresh_roc = p_threshold_rate_of_change


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict, centroids: list):

        inst = p_inst[-1].get_feature_data()
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
                    event = ClusterEnlargement(p_instances=p_inst)
                else:
                    self._spacial_sizes[cluster_id] = distance
            else:
                if (distance-self._ref_spacial_sizes[cluster_id]) > self._ref_spacial_sizes[cluster_id]*0.05:
                    self._ref_spacial_sizes[cluster_id] = distance
                    self._spacial_sizes[cluster_id] = self._ref_spacial_sizes[cluster_id]
                    event = ClusterEnlargement(p_instances=p_inst)
                else:
                    self._spacial_sizes[cluster_id] = distance




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterVelocityChangeDetector(AnomalyDetectorCB):
    """
    This is the class for detecting change in velocity of clusters.

    """
    C_PROPERTIY_DEFINITIONS : PropertyDefinitions = [ ['centroid', 2, Property]]

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
        
        for x in self.C_PROPERTIY_DEFINITIONS:
            if x not in self.C_REQ_CLUSTER_PROPERTIES:
                self.C_REQ_CLUSTER_PROPERTIES.append(x)

        unknown_prop = self._clusterer.align_cluster_properties(p_properties=self.C_REQ_CLUSTER_PROPERTIES)

        if len(unknown_prop) >0:
            raise RuntimeError("The following cluster properties need to be provided by the clusterer: ", unknown_prop)
        

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict, centroids: list):
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
    C_PROPERTIY_DEFINITIONS : PropertyDefinitions = [ ['density', 2, Property]]

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

        for x in self.C_PROPERTIY_DEFINITIONS:
            if x not in self.C_REQ_CLUSTER_PROPERTIES:
                self.C_REQ_CLUSTER_PROPERTIES.append(x)

        unknown_prop = self._clusterer.align_cluster_properties(p_properties=self.C_REQ_CLUSTER_PROPERTIES)

        if len(unknown_prop) >0:
            raise RuntimeError("The following cluster properties need to be provided by the clusterer: ", unknown_prop)

        self._thresh_ul = p_threshold_upper_limit
        self._thresh_ll = p_threshold_lower_limit
        self._thresh_det = p_threshold_detection
        self._thresh_roc = p_threshold_rate_of_change


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict, centroids: list):

        pass



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterSizeChangeDetector(AnomalyDetectorCB):
    """
    This is the class for detecting change in weight of clusters.

    """
    C_PROPERTIY_DEFINITIONS : PropertyDefinitions = [ ['size', 2, Property]]


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_clusterer : ClusterAnalyzer = None,
                 p_threshold : int = 1000,
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
        
        for x in self.C_PROPERTIY_DEFINITIONS:
            if x not in self.C_REQ_CLUSTER_PROPERTIES:
                self.C_REQ_CLUSTER_PROPERTIES.append(x)

        unknown_prop = self._clusterer.align_cluster_properties(p_properties=self.C_REQ_CLUSTER_PROPERTIES)

        if len(unknown_prop) >0:
            raise RuntimeError("The following cluster properties need to be provided by the clusterer: ", unknown_prop)

        
        self._threshold = p_threshold


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict):
        
        clusters = self._clusterer.get_clusters()

        ano_scores = []
        
        for x in clusters.keys():
            self._weights[x] = len(clusters[x])
            if self._weights > self._threshold:
                ano_scores.append(-1)
            else:
                ano_scores.append(0)




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterRelativeSizeChangeDetector(AnomalyDetectorCB):
    """
    This is the class for detecting change in weight of clusters.

    """
    C_PROPERTIY_DEFINITIONS : PropertyDefinitions = [ ['size', 2, Property]]

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_clusterer : ClusterAnalyzer = None,
                 p_scale : int = 1,
                 p_window : int = 100,
                 p_threshold : int = 3,
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
        
        for x in self.C_PROPERTIY_DEFINITIONS:
            if x not in self.C_REQ_CLUSTER_PROPERTIES:
                self.C_REQ_CLUSTER_PROPERTIES.append(x)

        unknown_prop = self._clusterer.align_cluster_properties(p_properties=self.C_REQ_CLUSTER_PROPERTIES)

        if len(unknown_prop) >0:
            raise RuntimeError("The following cluster properties need to be provided by the clusterer: ", unknown_prop)


        self._scale = p_scale
        self.window_size = p_window
        self.threshold = p_threshold
        self.cluster_sizes = {}
        self.total_size = 0
        self.history = {}


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict, center: float, centroids: list):

        clusters = self._clusterer.get_clusters()
        total_weight = 0
        for x in clusters.keys():
            total_weight += len(clusters[x])

        
        # Calculate relative size
        for x in clusters.keys():
            self._rel_weights[x] = len(clusters[x])/total_weight if total_weight>0 else 0

        # Update history
        for x in clusters.keys():
            if x not in self.history.keys():
                self.history[x] = []
            self.history[x].append(self._rel_weights[x])
            # Maintain a fixed size window
            if len(self.history[x]) > self.window_size:
                self.history[x].pop(0)

        z_score = []
        for cluster_id in self.history.keys():
            mean = np.mean(self.history[cluster_id])
            std = np.std(self.history[cluster_id])
            current_relative_size = self.history[cluster_id][-1]
            z_score.append((current_relative_size - mean) / std if std > 0 else 0)

        for x in range(len(z_score)):
            if abs(z_score[x]) > self.threshold:
                z_score[x] = -1
            else:
                z_score[x] = 0
        return z_score
        



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NewClusterDetector(AnomalyDetectorCB):
    """
    This is the class for detecting new clusters.

    """
    C_PROPERTIY_DEFINITIONS : PropertyDefinitions = [ ['centroid', 2, Property]]

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_clusterer : ClusterAnalyzer = None,
                 p_threshold : float = 0.1,
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
        
        for x in self.C_PROPERTIY_DEFINITIONS:
            if x not in self.C_REQ_CLUSTER_PROPERTIES:
                self.C_REQ_CLUSTER_PROPERTIES.append(x)

        unknown_prop = self._clusterer.align_cluster_properties(p_properties=self.C_REQ_CLUSTER_PROPERTIES)

        if len(unknown_prop) >0:
            raise RuntimeError("The following cluster properties need to be provided by the clusterer: ", unknown_prop)

        self.previous_centroids = []
        self.distance_threshold = p_threshold
        

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict, centroids: list):
        
        clusters = self._clusterer.get_clusters()
        if len(clusters) > self._num_clusters:
            print((len(clusters)-self._num_clusters), "new clusters appeared.")
            event = NewClusterAppearance()
        self._num_clusters = len(clusters)

        centroids = [tuple(centroid) for centroid in centroids]
        if not self.previous_centroids:
            self.previous_centroids = centroids
            return {"new_clusters": centroids, "split_clusters": [], "merged_clusters": []}

        # Calculate distances between old and new centroids
        distance_matrix = cdist(self.previous_centroids, centroids)

        # Find which centroids are considered the same (below distance threshold)
        matched_old = set()
        matched_new = set()
        for i, row in enumerate(distance_matrix):
            for j, distance in enumerate(row):
                if distance <= self.distance_threshold:
                    matched_old.add(i)
                    matched_new.add(j)

        new_clusters = [centroids[j] for j in range(len(centroids)) if j not in matched_new]
        split_clusters = [self.previous_centroids[i] for i in range(len(self.previous_centroids)) if i not in matched_old]
        merged_clusters = [centroids[j] for j in matched_new if list(distance_matrix[:, j]).count(distance_matrix[:, j].min()) > 1]

        self.previous_centroids = centroids
        return {"new_clusters": new_clusters, "split_clusters": split_clusters, "merged_clusters": merged_clusters}





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterDisappearanceDetector(AnomalyDetectorCB):
    """
    This is the class for detecting the disappearences of clusters.

    """
    C_PROPERTIY_DEFINITIONS : PropertyDefinitions = [ ['centroid', 2, Property]]

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_clusterer : ClusterAnalyzer = None,
                 p_threshold : float = 0.1,
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

        for x in self.C_PROPERTIY_DEFINITIONS:
            if x not in self.C_REQ_CLUSTER_PROPERTIES:
                self.C_REQ_CLUSTER_PROPERTIES.append(x)

        unknown_prop = self._clusterer.align_cluster_properties(p_properties=self.C_REQ_CLUSTER_PROPERTIES)

        if len(unknown_prop) >0:
            raise RuntimeError("The following cluster properties need to be provided by the clusterer: ", unknown_prop)


        self.previous_centroids = []
        self.distance_threshold = p_threshold

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict, center: float, centroids: list):
        
        clusters = self._clusterer.get_clusters()
        if len(clusters) < self._num_clusters:
            print((self._num_clusters-len(clusters)), "clusters disappeared")
            event = ClusterDisappearence()
        self._num_clusters = len(clusters)

        centroids = [tuple(centroid) for centroid in centroids]

        # Calculate distances between old and new centroids
        distance_matrix = cdist(self.previous_centroids, centroids)

        # Find which centroids are considered the same (below distance threshold)
        matched_old = set()
        matched_new = set()
        for i, row in enumerate(distance_matrix):
            for j, distance in enumerate(row):
                if distance <= self.distance_threshold:
                    matched_old.add(i)
                    matched_new.add(j)

        merged_clusters = []
        disappeared_clusters = [self.previous_centroids[i] for i in range(len(self.previous_centroids)) if i not in matched_old]

        for j in matched_new:
            if list(distance_matrix[:, j]).count(min(distance_matrix[:, j])) > 1:
                merged_clusters.append(centroids[j])

        self.previous_centroids = centroids
        return {
            "merged_clusters": merged_clusters,
            "disappeared_clusters": disappeared_clusters
        }

