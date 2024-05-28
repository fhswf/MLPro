## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.cb_detectors
## -- Module  : disappearance_detector.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -- 2024-05-22  1.2.1     SK       Refactoring
## -- 2024-05-28  1.2.2     SK       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.2 (2024-05-28)

This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.tasks.anomalydetectors.cb_detectors.basics import AnomalyDetectorCB
from MLPro.src.mlpro.oa.streams.tasks.anomalydetectors.anomalies import *
from mlpro.bf.streams import Instance, InstDict
from mlpro.bf.math.properties import *
from mlpro.oa.streams.tasks.clusteranalyzers.basics import ClusterAnalyzer
from scipy.spatial.distance import cdist




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
            event = ClusterDisappearance()
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

