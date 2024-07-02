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
## -- 2024-06-22  1.1.1     SK       Bug Fix
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.1 (2024-06-22)

This module provides templates for anomaly detection to be used in the context of online adaptivity.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.tasks.anomalydetectors.cb_detectors.basics import AnomalyDetectorCB
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased.size import ClusterSizeVariation
from mlpro.oa.streams.tasks.clusteranalyzers.basics import ClusterAnalyzer
from mlpro.bf.streams import Instance, InstDict
from mlpro.bf.math.properties import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterSizeChangeDetector(AnomalyDetectorCB):
    """
    This is the class for detecting change in size/weight of clusters.

    """
    C_PROPERTIY_DEFINITIONS : PropertyDefinitions = [ ( 'size', 0, False, Property )]


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_clusterer : ClusterAnalyzer = None,
                 p_size_thresh : float = None,
                 p_size_upper_thresh : float = None,
                 p_size_lower_thresh : float = None,
                 p_roc_size_thresh : float = 0.1,
                 p_relative_size_change : bool = False,
                 p_step_rate = 1,
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
        
        for x in self.C_PROPERTIY_DEFINITIONS:
            if x not in self.C_REQ_CLUSTER_PROPERTIES:
                self.C_REQ_CLUSTER_PROPERTIES.append(x)

        unknown_prop = self._clusterer.align_cluster_properties(p_properties=self.C_REQ_CLUSTER_PROPERTIES)

        if len(unknown_prop) >0:
            raise RuntimeError("The following cluster properties need to be provided by the clusterer: ", unknown_prop)

        self._rel_size_change = p_relative_size_change
        self._thresh_u      = p_size_upper_thresh
        self._thresh_l      = p_size_lower_thresh
        self._thresh        = p_size_thresh
        self._thresh_roc    = p_roc_size_thresh

        self._prev_sizes      = {}
        #self._prev_roc_sizes  = {}
        self._size_thresh     = {}

        self._step_rate = p_step_rate
        self._init_skip = p_initial_skip

        self._size_buffer     = {}

        self._sizes = {}
        self._count = {}
        self._distr_bias = {}


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst : InstDict):
        new_instances = []
        for inst_id, (inst_type, inst) in sorted(p_inst.items()):
            new_instances.append(inst)

        clusters = self._clusterer.get_clusters()

        affected_clusters = {}

        if self._thresh_u != None:
            for x in clusters.keys():
                if clusters[x].size.value >= self._thresh_u:
                    affected_clusters[x] = clusters[x]
        if self._thresh_l != None:
            for x in clusters.keys():
                if clusters[x].size.value <= self._thresh_l:
                    affected_clusters[x] = clusters[x]

        if self._size_thresh:
            for x in clusters.keys():
                if x not in self._prev_sizes.keys():
                    self._prev_sizes[x] = clusters[x].size.value
                    self._size_thresh[x] = clusters[x].size.value * self._size_thresh
            
                if (self._prev_sizes[x]-clusters[x].size.value) >= self._size_thresh[x]:
                    affected_clusters[x] = clusters[x]
                    self._prev_sizes[x] = clusters[x].size.value
                    self._size_thresh[x] = clusters[x].size.value * self._size_thresh
                elif (clusters[x].size.value-self._prev_sizes[x]) >= self._size_thresh[x]:
                    affected_clusters[x] = clusters[x] 
                    self._prev_sizes[x] = clusters[x].size.value
                    self._size_thresh[x] = clusters[x].size.value * self._size_thresh

        if self._rel_size_change:
            
            for x in clusters.keys():
                if x not in self._sizes.keys():
                    self._sizes[x] = int(clusters[x].size.value)
                    self._count[x] = 1
                    self._distr_bias[x] = 0

                else:
                    if int(clusters[x].size.value) == self._sizes[x]:
                        self._count[x] += 1

                    else:
                        self._distr_bias[x] = self._count[x]
                        self._count[x] = 1
                        self._sizes[x] = int(clusters[x].size.value)
            
            print(self._distr_bias)



        """if x not in self._size_buffer.keys():
                self._size_buffer[x] = []
            self._size_buffer[x].append(clusters[x].size.value)
            if len(self._size_buffer[x]) > (self._step_rate+1):
                self._size_buffer[x].pop(0)

        event = ClusterSizeVariation(p_id = self._get_next_anomaly_id,
                                     p_instances=[inst],
                                     p_clusters=affected_clusters)"""
        
        