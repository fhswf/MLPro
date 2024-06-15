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
## -- 2024-05-28  1.2.1     SK       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.1 (2024-05-28)

This module provides cluster disappearance detector algorithm.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.tasks.anomalydetectors.cb_detectors.basics import AnomalyDetectorCB
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased.disappearance import ClusterDisappearance
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
                 p_age_threshold : int = None,
                 p_size_threshold : int = None,
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

        self._current_clusters = {}
        self._age_thresh = p_age_threshold
        self._size_thresh = p_size_threshold

        if self._age_thresh != None:
            self.C_PROPERTIY_DEFINITIONS.append(['age', 0, Property])

        if self._size_thresh != None:
            self.C_PROPERTIY_DEFINITIONS.append(['size', 0, Property])

        for x in self.C_PROPERTIY_DEFINITIONS:
            if x not in self.C_REQ_CLUSTER_PROPERTIES:
                self.C_REQ_CLUSTER_PROPERTIES.append(x)

        unknown_prop = self._clusterer.align_cluster_properties(p_properties=self.C_REQ_CLUSTER_PROPERTIES)

        if len(unknown_prop) >0:
            raise RuntimeError("The following cluster properties need to be provided by the clusterer: ", unknown_prop)

        self._current_clusters = {}


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict, center: float, centroids: list):
        

        inst = []

        for inst_id, (inst_id, inst_1) in sorted(p_inst.items()):
            inst = inst_1

        clusters = self._clusterer.get_clusters()

        missing_clusters = {}

        if len(clusters) < len(self._current_clusters):
            
            for x in self._current_clusters:
                if x not in clusters:
                    missing_clusters[x] = self._current_clusters[x]

        if self._age_thresh != None:
            for x in clusters.keys():
                if clusters[x].age.value <= self._age_thresh:
                    missing_clusters[x] = clusters[x]

        if self._size_thresh != None:
            for x in clusters.keys():
                if clusters[x].age.value <= self._size_thresh:
                    missing_clusters[x] = clusters[x]

        event = ClusterDisappearance(p_id = self._get_next_anomaly_id,
                                     p_instances=[inst],
                                     p_clusters=missing_clusters)

        self._current_clusters = clusters
