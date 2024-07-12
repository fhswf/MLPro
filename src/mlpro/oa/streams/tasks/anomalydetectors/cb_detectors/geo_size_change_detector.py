## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.cb_detectors
## -- Module  : geo_size_change_detector.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-08  0.0.0     SK       Creation
## -- 2023-09-12  1.0.0     SK       Release
## -- 2024-04-10  1.1.0     DA/SK    Refactoring
## -- 2024-06-22  1.1.1     SK       Bug Fix
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.1 (2024-06-22)

This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.tasks.anomalydetectors.cb_detectors.basics import AnomalyDetectorCB
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased.enlargement import ClusterEnlargement
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased.shrinkage import ClusterShrinkage
from mlpro.oa.streams.tasks.clusteranalyzers.basics import ClusterAnalyzer
from mlpro.bf.streams import Instance, InstDict
from mlpro.bf.math.properties import *
import time



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterGeometricSizeChangeDetector(AnomalyDetectorCB):
    """
    This is the class for detecting change of spatial size of clusters.

    """
    
    C_PROPERTY_DEFINITIONS : PropertyDefinitions = [ ( 'size_geo', 0, False, Property )]

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_clusterer : ClusterAnalyzer = None,
                 p_geo_size_thresh : float = 0.01,
                 p_roc_geo_size_thresh : float = 0.005,
                 p_initial_skip : int = 1,
                 p_buffer_size: int = 5,
                 p_window_size: int = 10,
                 p_time_calculation: bool = False,
                 p_rel_threshold : bool = False,
                 p_geo_size_upper_thresh : float = None,
                 p_geo_size_lower_thresh : float = None,
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

        #if len(unknown_prop) >0:
        #    raise RuntimeError("The following cluster properties need to be provided by the clusterer: ", unknown_prop)
        
        self._thresh_u      = p_geo_size_upper_thresh
        self._thresh_l      = p_geo_size_lower_thresh
        self._thresh        = p_geo_size_thresh
        self._roc_thresh    = p_roc_geo_size_thresh

        self._rel_thresh = p_rel_threshold
        self._visualize = p_visualize
        self._init_skip = p_initial_skip
        self._count = 1

        self._time_calculation = p_time_calculation
        self._buffer_size = p_buffer_size
        self._window_size = p_window_size

        self._geo_size_history = {}
        self._avg_geo_size_history = {}
        self._time_history = {}
        self._current_state = {}


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict):
        new_instances = []
        for inst_id, (inst_type, inst) in sorted(p_inst.items()):
            new_instances.append(inst)

        clusters = self._clusterer.get_clusters()
        
        if self._rel_thresh:
            thresh = self._calculate_threshold(clusters=clusters, thresh=self._thresh)
            roc_thresh = self._calculate_threshold(clusters=clusters, thresh=self._roc_thresh)
        else:
            thresh = self._thresh
            roc_thresh = self._roc_thresh

        affected_clusters_shrinkage = {}
        affected_clusters_enlargement = {}

        for id in clusters.keys():
            current_time = time.time()
            if id not in self._geo_size_history.keys():
                self._geo_size_history[id] = [clusters[id].size_geo.value]
            else:
                self._geo_size_history[id].append(clusters[id].size_geo.value)

            if len(self._geo_size_history[id]) > self._buffer_size:
                self._geo_size_history[id].pop(0)

            filtered_sizes = [size for size in self._geo_size_history[id] if size is not None]
            
            if not filtered_sizes:
                avg_size = 0
            else:
                avg_size = sum(filtered_sizes) / len(filtered_sizes)

            if id not in self._avg_geo_size_history.keys():
                self._avg_geo_size_history[id] = [avg_size]
                self._time_history[id] = [current_time]
            else:
                self._avg_geo_size_history[id].append(avg_size)
                self._time_history[id].append(current_time)

            if len(self._avg_geo_size_history[id]) > self._window_size:
                self._avg_geo_size_history[id].pop(0)
                self._time_history[id].pop(0)

            if self._thresh_u:
                if clusters[id].size_geo.value != None:
                    if clusters[id].size_geo.value >= self._thresh_u:
                        affected_clusters_enlargement[id] = clusters[id]
            if self._thresh_l:
                if clusters[id].size_geo.value != None:
                    if clusters[id].size_geo.value <= self._thresh_l:
                        affected_clusters_shrinkage[id] = clusters[id]

            if len(self._avg_geo_size_history[id]) == 1:
            # Only one data point, not enough to determine any change
                self._current_state[id] = "NC"

            elif len(self._avg_geo_size_history[id]) == 2:
                # Only two data points, can determine if there's an initial change
                if self._time_calculation:
                    time_diff = self._time_history[id][1] - self._time_history[id][0]
                    first_diff = (self._avg_geo_size_history[id][1] - self._avg_geo_size_history[id][0]) / time_diff if time_diff != 0 else 0
                else:
                    first_diff = self._avg_geo_size_history[id][1] - self._avg_geo_size_history[id][0]

                if first_diff > thresh:
                    current_state = "LI"
                    if current_state != self._current_state[id]:
                        affected_clusters_enlargement[id] = clusters[id]
                        self._current_state[id] = current_state
                elif first_diff < -thresh:
                    current_state = "LD"
                    if current_state != self._current_state[id]:
                        affected_clusters_shrinkage[id] = clusters[id]
                        self._current_state[id] = current_state
                else:
                    current_state = "NC"

            else:
                # Calculate first differences
                if self._time_calculation:
                    first_diff = [(self._avg_geo_size_history[id][i+1] - self._avg_geo_size_history[id][i]) / (self._time_history[id][i+1] - self._time_history[id][i]) if (self._time_history[id][i+1] - self._time_history[id][i]) != 0 else 0 for i in range(len(self._avg_geo_size_history[id])-1)]
                else:
                    first_diff = [self._avg_geo_size_history[id][i+1] - self._avg_geo_size_history[id][i] for i in range(len(self._avg_geo_size_history[id])-1)]

                # Calculate second differences if enough data points are available
                if len(self._avg_geo_size_history[id]) > 2:
                    second_diff = [first_diff[i+1] - first_diff[i] for i in range(len(first_diff)-1)]
                else:
                    second_diff = []

                # Determine the current state
                if all(d > thresh for d in first_diff):
                    current_state = "LI"
                    if current_state != self._current_state[id]:
                        affected_clusters_enlargement[id] = clusters[id]
                        self._current_state[id] = current_state
                elif all(d < -thresh for d in first_diff):
                    current_state = "LD"
                    if current_state != self._current_state[id]:
                        affected_clusters_shrinkage[id] = clusters[id]
                        self._current_state[id] = current_state
                elif any(d > thresh for d in first_diff) and any(abs(d2) > roc_thresh for d2 in second_diff):
                    current_state = "VI"
                    if current_state != self._current_state[id]:
                        affected_clusters_enlargement[id] = clusters[id]
                        self._current_state[id] = current_state
                elif any(d < -thresh for d in first_diff) and any(abs(d2) > roc_thresh for d2 in second_diff):
                    current_state = "VD"
                    if current_state != self._current_state[id]:
                        affected_clusters_shrinkage[id] = clusters[id]
                        self._current_state[id] = current_state
                else:
                    current_state = "NC"

        if self._count <= self._init_skip:
            self._count+= 1
            return

        if len(affected_clusters_shrinkage) != 0:
            anomaly = ClusterShrinkage(p_id = self._get_next_anomaly_id,
                                         p_instances=new_instances,
                                         p_clusters=affected_clusters_shrinkage,
                                         p_visualize=self._visualize)
            self._raise_anomaly_event(p_anomaly=anomaly)

        if len(affected_clusters_enlargement) != 0:
            anomaly = ClusterEnlargement(p_id = self._get_next_anomaly_id,
                                         p_instances=new_instances,
                                         p_clusters=affected_clusters_enlargement,
                                         p_visualize=self._visualize)
            self._raise_anomaly_event(p_anomaly=anomaly)


## -------------------------------------------------------------------------------------------------
    def _calculate_threshold(self, clusters, thresh):
            
        n = 0.0
        s = 0.0
        for x in clusters.keys():
            if clusters[x].size_geo.value != None:
                if clusters[x].size_geo.value > 0.0:
                    n += 1
                    s += abs(float(1/clusters[x].size_geo.value))

        if s != 0.0:
            return  ((n * thresh/100) / s)
        else: return 0.0

        
