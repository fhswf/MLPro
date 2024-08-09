## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.cb_detectors
## -- Module  : new_cluster_detector.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2024-04-10  1.1.0     DA/SK    Refactoring
## -- 2024-05-28  1.2.0     SK       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2024-05-28)

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
    C_PROPERTY_DEFINITIONS : PropertyDefinitions = []

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
        
        self._p_visualize = p_visualize

        self._prev_clusters = {}
        self._visualize = p_visualize
        

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst: InstDict):
        new_instances = []
        for inst_id, (inst_type, inst) in sorted(p_inst.items()):
            new_instances.append(inst)

        clusters = self._clusterer.get_clusters()

        if len(clusters) > len(self._prev_clusters):
            new_clusters = {}
            for x in clusters.keys():
                if x not in self._prev_clusters.keys():
                    new_clusters[x] = clusters[x]

            if new_clusters:  # Only raise an event if there are new clusters
                anomaly = NewClusterAppearance(p_id=self._get_next_anomaly_id,
                                               p_instances=new_instances,
                                               p_clusters=new_clusters,
                                               p_det_time=str(inst.get_tstamp()),
                                               p_visualize=self._p_visualize)
                self._raise_anomaly_event(anomaly)

            self._prev_clusters.update(new_clusters) 
