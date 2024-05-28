## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.cb_detectors
## -- Module  : geo_size_change_detector.py
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

