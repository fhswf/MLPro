## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.cb_detectors
## -- Module  : size_change_detector.py
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

This module provides cluster size change detector algorithm.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.tasks.anomalydetectors.cb_detectors.basics import AnomalyDetectorCB
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased.size import ClusterSizeVariation
from mlpro.oa.streams.tasks.clusteranalyzers.basics import ClusterAnalyzer
from mlpro.bf.streams import Instance, InstDict
from mlpro.bf.math.properties import *
import time




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterSizeChangeDetector(AnomalyDetectorCB):
    """
    This is the class for detecting change in size/weight of clusters.

    """
    C_PROPERTY_DEFINITIONS : PropertyDefinitions = [( 'size', 0, False, Property )]

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_clusterer : ClusterAnalyzer = None,
                 p_size_thresh_factor : float = False,
                 p_roc_size_thresh_factor : float = False,
                 p_initial_skip : int = 1,
                 p_ema_alpha : float = 0.7,
                 p_window_size: int = 10,
                 p_with_time_calculation: bool = False,
                 p_relative_thresh : bool = False,
                 p_size_upper_thresh : float = None,
                 p_size_lower_thresh : float = None,
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
        
        self._thresh_u      = p_size_upper_thresh
        self._thresh_l      = p_size_lower_thresh
        self._thresh        = {"thresh":p_size_thresh_factor}
        self._roc_thresh    = {"thresh":p_roc_size_thresh_factor}
        self._ema = p_ema_alpha
        self._rel_thresh = p_relative_thresh
        self._visualize = p_visualize
        self._init_skip = p_initial_skip
        self._count = 1

        self._time_calculation = p_with_time_calculation
        self._window_size = p_window_size

        self._avg_size_history = {}
        self._time_history = {}
        self._current_state = {}
        self._buffer = {}

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict):
        new_instances = []
        for inst_id, (inst_type, inst) in sorted(p_inst.items()):
            new_instances.append(inst)

        clusters = self._clusterer.get_clusters()
        current_time = time.time()

        affected_clusters = {}

        for id in clusters.keys():
            if clusters[id].size.value == None:
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

            if id not in self._avg_size_history.keys():
                self._size_history[id] = []
                self._avg_size_history[id] = [clusters[id].size.value]
                self._time_history[id] = [current_time]
                self._update_history(id, clusters[id].size.value ,current_time)
            #self._size_history[id].append(clusters[id].size.value)

            self._detect_anomalies(id, clusters[id], affected_clusters,
                                   thresh, roc_thresh, current_time)
            
            self._update_history(id, clusters[id].size.value ,current_time)
            
            if id in affected_clusters.keys():
                    self._update_threshold(id, clusters)

        if self._count <= self._init_skip:
            self._count+= 1
            return

        if len(affected_clusters) != 0:
            anomaly = ClusterSizeVariation(p_id = self._get_next_anomaly_id,
                                         p_instances=[inst],
                                         p_clusters=affected_clusters,
                                         p_visualize=self._visualize)
            self._raise_anomaly_event(p_anomaly=anomaly)


## -------------------------------------------------------------------------------------------------
    def _update_history(self, id, value, current_time):
        avg_size = self[id][-1]*(1-self._ema) + value*self._ema

        self[id].append(avg_size)
        self._time_history[id].append(current_time)

        if len(self[id]) > self._window_size:
            self._avg_size_history[id].pop(0)
            self._time_history[id].pop(0)


## -------------------------------------------------------------------------------------------------
    def _detect_anomalies(self, id, cluster, affected_clusters, thresh, roc_thresh, current_time):
        if self._thresh_u and cluster.size.value != None and cluster.size.value >= self._thresh_u:
            affected_clusters[id] = cluster
            
        if self._thresh_l and cluster.size.value != None and cluster.size.value <= self._thresh_l:
            affected_clusters[id] = cluster

        if len(self._avg_size_history[id]) < 3:
            self._current_state[id] = "NC"
        else:
            self._state_change_detection(id, cluster, affected_clusters, thresh, roc_thresh, current_time)


## -------------------------------------------------------------------------------------------------
    def _state_change_detection(self, id, cluster, affected_clusters, thresh, roc_thresh, current_time):
        # Calculate first differences
        if self._time_calculation:
            first_diff = [(self._avg_size_history[id][i+1] - self._avg_size_history[id][i]) / (self._time_history[id][i+1] - self._time_history[id][i]) if (self._time_history[id][i+1] - self._time_history[id][i]) != 0 else 0 for i in range(len(self._avg_size_history[id])-1)]
            time_diff = current_time - self._time_history[id][-1]
            diff = (cluster.size.value - self._avg_size_history[id][-1]) / time_diff if time_diff != 0 else 0.0
            first_diff.append(diff)
        else:
            first_diff = [self._avg_size_history[id][i+1] - self._avg_size_history[id][i] for i in range(len(self._avg_size_history[id])-1)]
            diff = cluster.size.value - self._avg_size_history[id][-1]
            first_diff.append(diff)


        # Calculate second differences if enough data points are available
        if roc_thresh:
            if len(self._avg_size_history[id]) > 2:
                second_diff = [first_diff[i+1] - first_diff[i] for i in range(len(first_diff)-1)]
            else:
                second_diff = []

        #if self._rel_change_thresh:
        #    if abs(diff) > self._rel_change_thresh:
        #        affected_clusters[id] = cluster

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
            affected_clusters[id] = cluster
            self._current_state[id] = current_state
            

## -------------------------------------------------------------------------------------------------
    def _update_threshold(self, id, clusters):
        if clusters[id].size.value > 0:
            if self._rel_thresh:    
                n = 0.0
                s = 0.0
                for x in clusters.keys():
                    if clusters[x].size.value != None:
                        if clusters[x].size.value > 0.0:
                            n += 1
                            s += abs(float(1/clusters[x].size.value))

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
                    self._thresh[id] = float(clusters[id].size.value)*self._thresh["thresh"]
                if self._roc_thresh["thresh"]:
                    self._roc_thresh[id] = float(clusters[id].size.value)*self._roc_thresh["thresh"]

  
