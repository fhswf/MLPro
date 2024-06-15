## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.cb_detectors
## -- Module  : new_cluster_detector.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2023-11-21  1.0.1     SK       Time Stamp update
## -- 2024-02-25  1.1.0     SK       Visualisation update
## -- 2024-04-10  1.2.0     DA/SK    Refactoring
## -- 2024-05-28  1.2.1     SK       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.1 (2024-05-28)

This module provides new cluster detector algorithm.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.tasks.anomalydetectors.cb_detectors.basics import AnomalyDetectorCB
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased.new_cluster import NewClusterAppearance
from mlpro.oa.streams.tasks.clusteranalyzers.basics import ClusterAnalyzer
from mlpro.bf.streams import Instance, InstDict
from mlpro.bf.math.properties import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NewClusterDetector(AnomalyDetectorCB):
    """
    This is the class for detecting new clusters.

    """
    C_PROPERTIY_DEFINITIONS : PropertyDefinitions = []

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

        if len(unknown_prop) > 0:
            raise RuntimeError("The following cluster properties need to be provided by the clusterer: ", unknown_prop)

        self._current_clusters = {}
        

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict):

        inst = []

        for inst_id, (inst_id, inst_1) in sorted(p_inst.items()):
            inst = inst_1

        clusters = self._clusterer.get_clusters()
        if len(clusters) > len(self._current_clusters):
            new_clusters = {}
            for x in clusters:
                if x not in self._current_clusters:
                    new_clusters[x] = clusters[x]
            event = NewClusterAppearance(p_id = self._get_next_anomaly_id,
                                         p_instances=[inst],
                                         p_clusters=new_clusters)
            
        self._current_clusters = clusters

        """centroids = [tuple(centroid) for centroid in centroids]
        if not self._previous_centroids:
            self._previous_centroids = centroids
            return {"new_clusters": centroids, "split_clusters": [], "merged_clusters": []}

        # Calculate distances between old and new centroids
        distance_matrix = cdist(self._previous_centroids, centroids)

        # Find which centroids are considered the same (below distance threshold)
        matched_old = set()
        matched_new = set()
        for i, row in enumerate(distance_matrix):
            for j, distance in enumerate(row):
                if distance <= self._distance_threshold:
                    matched_old.add(i)
                    matched_new.add(j)

        new_clusters = [centroids[j] for j in range(len(centroids)) if j not in matched_new]
        split_clusters = [self._previous_centroids[i] for i in range(len(self._previous_centroids)) if i not in matched_old]
        merged_clusters = [centroids[j] for j in matched_new if list(distance_matrix[:, j]).count(distance_matrix[:, j].min()) > 1]

        self._previous_centroids = centroids
        return {"new_clusters": new_clusters, "split_clusters": split_clusters, "merged_clusters": merged_clusters}"""


