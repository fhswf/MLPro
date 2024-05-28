## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.cb_detectors
## -- Module  : rel_size_change_detector.py
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
from MLPro.src.mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased import *
from mlpro.oa.streams.tasks.clusteranalyzers.basics import ClusterAnalyzer
from mlpro.bf.streams import Instance, InstDict
import numpy as np
from mlpro.bf.math.properties import *




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
        
