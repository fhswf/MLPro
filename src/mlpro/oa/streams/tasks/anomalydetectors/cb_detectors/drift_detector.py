## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.cb_detectors
## -- Module  : drift_detector.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2024-04-10  1.1.0     DA/SK    Refactoring
## -- 2024-06-20  1.1.1     SK       Bug Fix
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.1 (2024-06-20)

This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.tasks.anomalydetectors.cb_detectors.basics import AnomalyDetectorCB
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased.drift import ClusterDrift
from mlpro.oa.streams.tasks.clusteranalyzers.basics import ClusterAnalyzer
from mlpro.bf.streams import Instance, InstDict
from mlpro.bf.math.properties import *
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties.centroid import cprop_center_geo2
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import cprop_size2




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterDriftDetector(AnomalyDetectorCB):
    """
    This is the class for detecting change in velocity of clusters.

    """
    C_PROPERTY_DEFINITIONS : PropertyDefinitions = [ cprop_center_geo2,
                                                     cprop_size2]

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_clusterer : ClusterAnalyzer = None,
                 p_velocity_threshold : float = 4.0,
                 p_step_rate = 10,
                 p_initial_skip : int = 1,
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
        
        for x in self.C_PROPERTY_DEFINITIONS:
            if x not in self.C_REQ_CLUSTER_PROPERTIES:
                self.C_REQ_CLUSTER_PROPERTIES.append(x)

        unknown_prop = self._clusterer.align_cluster_properties(p_properties=self.C_REQ_CLUSTER_PROPERTIES)

        #if len(unknown_prop) > 0:
        #    raise RuntimeError("The following cluster properties need to be provided by the clusterer: ", unknown_prop)
        
        self._cluster_centroids = {}
        self._thresh = p_velocity_threshold
        self._step_rate = p_step_rate
        self._count = 0
        self._init_skip = p_initial_skip
        self._centroid_history = {}
        self._prev_velocities = {}

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict):
        if (self._count >= self._init_skip):

            new_instances = []
            for inst_id, (inst_type, inst) in sorted(p_inst.items()):
                new_instances.append(inst)

            clusters = self._clusterer.get_clusters()

            drifting_clusters = {}

            for x in clusters.keys():
                if x not in self._centroid_history.keys():
                    self._centroid_history[x] = []

                self._centroid_history[x].append(clusters[x].centroid.value)
                if len(self._centroid_history[x]) == self._step_rate:
                    self._prev_velocities[x] = [(a - b)/self._step_rate for a, b in zip(self._centroid_history[x][-1], self._centroid_history[x][0])]

                if len(self._centroid_history[x]) > self._step_rate:
                    self._centroid_history[x].pop(0)

                    for y in range(len(clusters[x].centroid.value)):
                        velocity = self._centroid_history[x][-1][y] - self._centroid_history[x][0][y]
                        if abs(velocity) >= abs(self._prev_velocities[x][y] * self._thresh):
                            drifting_clusters[x] = clusters[x]
                            self._prev_velocities[x] = [(a - b)/self._step_rate for a, b in zip(self._centroid_history[x][-1], self._centroid_history[x][0])]
                            break

                        elif abs(velocity) <= abs(self._prev_velocities[x][y] / self._thresh):
                            drifting_clusters[x] = clusters[x]
                            self._prev_velocities[x] = [(a - b)/self._step_rate for a, b in zip(self._centroid_history[x][-1], self._centroid_history[x][0])]
                            break


            if len(drifting_clusters) != 0:

                anomaly = ClusterDrift(p_id = self._get_next_anomaly_id,
                                 p_instances=new_instances,
                                 p_clusters=drifting_clusters,
                                 p_det_time=str(inst.get_tstamp()))
                self._raise_anomaly_event(anomaly)
            
        self._count += 1
            


