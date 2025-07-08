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
## -- 2024-05-28  1.2.0     SK       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2024-05-28)

This module provides cluster geometrical size change detector algorithm.
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
                 p_geo_size_thresh_factor : float = False,
                 p_roc_geo_size_thresh_factor : float = False,
                 p_initial_skip : int = 1,
                 p_ema : float = 0.7,
                 p_window_size: int = 10,
                 p_with_time_calculation: bool = False,
                 p_relative_thresh : bool = False,
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
        
        self._thresh_u      = p_geo_size_upper_thresh
        self._thresh_l      = p_geo_size_lower_thresh
        self._thresh        = {"thresh":p_geo_size_thresh_factor}
        self._roc_thresh    = {"thresh":p_roc_geo_size_thresh_factor}
        self._ema = p_ema

        self._rel_thresh = p_relative_thresh
        self._visualize = p_visualize
        self._init_skip = p_initial_skip
        self._count = 1

        self._time_calculation = p_with_time_calculation
        self._window_size = p_window_size

        self._geo_size_history = {}
        self._time_history = {}
        self._current_state = {}


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict):
        new_instances = []
        for inst_id, (inst_type, inst) in sorted(p_inst.items()):
            new_instances.append(inst)

        clusters = self._clusterer.get_clusters()
        current_time = time.time()

        affected_clusters_shrinkage = {}
        affected_clusters_enlargement = {}

        for id in clusters.keys():
            if clusters[id].size_geo.value == None:
                continue

            if self._thresh["thresh"]:
                if id not in self._thresh.keys():
                    self._thresh[id] = self._thresh["thresh"]
                thresh = self._thresh[id]
            else:
                thresh = None
            if self._roc_thresh["thresh"]:
                if id not in self._roc_thresh.keys():
                    self._roc_thresh[id] = self._roc_thresh["thresh"]
                roc_thresh = self._roc_thresh[id]  
            else:
                roc_thresh = None

            if id not in self._geo_size_history.keys():
                self._geo_size_history[id] = [clusters[id].size_geo.value]
                self._time_history[id] = [current_time]
                self._update_history(id, clusters[id].size_geo.value ,current_time)

            self._detect_anomalies(id, clusters[id], affected_clusters_shrinkage, affected_clusters_enlargement,
                                   thresh, roc_thresh, current_time)
            
            self._update_history(id, clusters[id].size_geo.value ,current_time)
            
            if (id in affected_clusters_enlargement.keys()) or (id in affected_clusters_shrinkage.keys()):
                    self._update_threshold(id, clusters)

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
    def _update_history(self, id, value, current_time):
        avg_size = self._geo_size_history[id][-1]*(1-self._ema) + value*self._ema

        self._geo_size_history[id].append(avg_size)
        self._time_history[id].append(current_time)

        if len(self._geo_size_history[id]) > self._window_size:
            self._geo_size_history[id].pop(0)
            self._time_history[id].pop(0)


## -------------------------------------------------------------------------------------------------
    def _detect_anomalies(self, id, cluster, affected_clusters_shrinkage, affected_clusters_enlargement, thresh, roc_thresh, current_time):
        if self._thresh_u and cluster.size_geo.value != None and cluster.size_geo.value >= self._thresh_u:
            affected_clusters_enlargement[id] = cluster
            
        if self._thresh_l and cluster.size_geo.value != None and cluster.size_geo.value <= self._thresh_l:
            affected_clusters_shrinkage[id] = cluster

        if len(self._geo_size_history[id]) < 3:
            self._current_state[id] = "NC"
        else:
            self._state_change_detection(id, cluster, affected_clusters_shrinkage, affected_clusters_enlargement, thresh, roc_thresh, current_time)


## -------------------------------------------------------------------------------------------------
    def _state_change_detection(self, id, cluster, affected_clusters_shrinkage, affected_clusters_enlargement, thresh, roc_thresh, current_time):
        # Calculate first differences
        if self._time_calculation:
            first_diff = [(self._geo_size_history[id][i+1] - self._geo_size_history[id][i]) / (self._time_history[id][i+1] - self._time_history[id][i]) if (self._time_history[id][i+1] - self._time_history[id][i]) != 0 else 0 for i in range(len(self._geo_size_history[id])-1)]
            time_diff = current_time - self._time_history[id][-1]
            diff = (cluster.size_geo.value - self._geo_size_history[id][-1]) / time_diff if time_diff != 0 else 0.0
            first_diff.append(diff)
        else:
            first_diff = [self._geo_size_history[id][i+1] - self._geo_size_history[id][i] for i in range(len(self._geo_size_history[id])-1)]
            diff = cluster.size_geo.value - self._geo_size_history[id][-1]
            first_diff.append(diff)

        # Calculate second differences if enough data points are available
        if roc_thresh:
            if len(self._geo_size_history[id]) > 2:
                second_diff = [first_diff[i+1] - first_diff[i] for i in range(len(first_diff)-1)]
            else:
                second_diff = []

        current_state = None
        if thresh:
            if any(d > thresh for d in first_diff):
                current_state = "LI"
            elif any(d < -thresh for d in first_diff):
                current_state = "LD"

        if roc_thresh:
            if any(d2 > roc_thresh for d2 in second_diff):
                current_state = "VI"
            elif any(d2 < -roc_thresh for d2 in second_diff):
                current_state = "VD"

        if not current_state:
            current_state = "NC"

        if current_state != self._current_state[id]:
            if self._current_state[id] == "NC":
                if current_state in ["LI", "VI"]:
                    affected_clusters_enlargement[id] = cluster
                else:
                    affected_clusters_shrinkage[id] = cluster
            elif self._current_state[id] in ["LI", "VI"]:
                if current_state in ["NC", "VD", "LD"]:
                    affected_clusters_shrinkage[id] = cluster
            elif self._current_state[id] in ["LD", "VD"]:
                if current_state in ["NC", "VI", "LI"]:
                    affected_clusters_enlargement[id] = cluster
            self._current_state[id] = current_state
        

## -------------------------------------------------------------------------------------------------
    def _update_threshold(self, id, clusters):
        if clusters[id].size_geo.value > 0:
            if self._rel_thresh:    
                n = 0.0
                s = 0.0
                for x in clusters.keys():
                    if clusters[x].size_geo.value != None:
                        if clusters[x].size_geo.value > 0.0:
                            n += 1
                            s += abs(float(1/clusters[x].size_geo.value))

                if s != 0.0:
                    if self._thresh["thresh"]:
                        self._thresh[id] = ((n * self._thresh["thresh"]/100) / s)
                    if self._roc_thresh["thresh"]:
                        self._roc_thresh[id] = ((n * self._roc_thresh["thresh"]/100) / s)
                else:
                    if self._thresh["thresh"]:
                        self._thresh[id] = self._thresh["thresh"]
                    if self._roc_thresh["thresh"]:
                        self._roc_thresh[id] = self._roc_thresh["thresh"]

            else:
                if self._thresh["thresh"]:
                    self._thresh[id] = float(clusters[id].size_geo.value)*self._thresh["thresh"]
                if self._roc_thresh["thresh"]:
                    self._roc_thresh[id] = float(clusters[id].size_geo.value)*self._roc_thresh["thresh"]

            
