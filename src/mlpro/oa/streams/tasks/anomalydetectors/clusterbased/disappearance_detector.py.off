## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.cb_detectors
## -- Module  : disappearance_detector.py
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

This module provides cluster disappearance detector algorithm.
"""

from mlpro.oa.streams.basics import *
from mlpro.oa.streams.tasks.anomalydetectors.cb_detectors.basics import AnomalyDetectorCB
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased.disappearance import ClusterDisappearance
from mlpro.bf.streams import Instance, InstDict
from mlpro.bf.math.properties import *
from mlpro.oa.streams.tasks.clusteranalyzers.basics import ClusterAnalyzer




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ClusterDisappearanceDetector(AnomalyDetectorCB):
    """
    This is the class for detecting the disappearences of clusters.

    """
    C_PROPERTY_DEFINITIONS : PropertyDefinitions = []

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_clusterer : ClusterAnalyzer = None,
                 p_age_threshold : int = None,
                 p_size_threshold : int = None,
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

        self._age_thresh            = p_age_threshold
        self._size_thresh           = p_size_threshold
        self._prev_clusters         = {}
        self._deleted_clusters_age  = {}
        self._deleted_clusters_size = {}
        self._prev_age              = {}
        self._age_counter           = {}

        if self._age_thresh != None:
            self.C_PROPERTY_DEFINITIONS.append(('age', 0, False, Property))

        if self._size_thresh != None:
            self.C_PROPERTY_DEFINITIONS.append(('size', 0, False, Property))


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst: InstDict):
        new_instances = []
        for inst_id, (inst_type, inst) in sorted(p_inst.items()):
            new_instances.append(inst)

        clusters = self._clusterer.get_clusters()

        for x in self._deleted_clusters_age.keys():
            if x in clusters.keys():
                if float(clusters[x].age.value) > self._prev_age[x]:
                    del self._deleted_clusters_age[x]
                    self._prev_clusters[x] = clusters[x]
                else:
                    del clusters[x]
        
        for x in self._deleted_clusters_size.keys():
            if x in clusters.keys():
                if clusters[x].size.value > self._size_thresh:
                    del self._deleted_clusters_size[x]
                    self._prev_clusters[x] = clusters[x]
                else:
                    del clusters[x]

        missing_clusters = {}

        if len(clusters) < len(self._prev_clusters):
            
            for x in self._prev_clusters.keys():
                if x not in clusters.keys():
                    missing_clusters[x] = self._prev_clusters[x]
                    del self._prev_clusters[x]
                    
        if self._size_thresh != None:
            for x in clusters.keys():
                if clusters[x].size.value <= self._size_thresh:
                    missing_clusters[x] = clusters[x]
                    self._deleted_clusters_size[x] = clusters[x]
                    if x in self._prev_clusters.keys():
                        del self._prev_clusters[x]
        
        if self._age_thresh != None:
            for x in clusters.keys():
                if x not in self._prev_age.keys():
                    self._prev_age[x] = float(clusters[x].age.value)
                    self._age_counter[x] = 0
                else:
                    if (self._age_counter[x] % self._age_thresh) == 0:
                        if float(clusters[x].age.value) == self._prev_age[x]:
                            missing_clusters[x] = clusters[x]
                            self._deleted_clusters_age[x] = clusters[x]
                            if x in self._prev_clusters.keys():
                                del self._prev_clusters[x]
                        else:
                            self._prev_age[x] = float(clusters[x].age.value)
                self._age_counter[x] += 1


        if missing_clusters:
            anomaly = ClusterDisappearance(p_id=self._get_next_anomaly_id,
                                           p_instances=new_instances,
                                           p_clusters=missing_clusters,
                                           p_det_time=str(inst.get_tstamp()),
                                           p_visualize=self._visualize)

            self._raise_anomaly_event(anomaly)


