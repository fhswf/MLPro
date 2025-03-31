## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.cb_detectors
## -- Module  : point_group_anomaly_detector.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-28  0.0.0     DS       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2025-03-28)

This module provides cluster based point and group anomaly detector algorithm.
"""

from mlpro.bf.various import Log
from mlpro.bf.math.properties import *
from mlpro.bf.streams import InstDict, InstTypeNew
from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer, Cluster
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import cprop_size
from mlpro.oa.streams.tasks.anomalydetectors.clusterbased.basics import AnomalyDetectorCB
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased import AnomalyCB, PointAnomaly, GroupAnomaly, SpatialGroupAnomaly, TemporalGroupAnomaly


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorCBPAGA(AnomalyDetectorCB):
    """
    Implementation of a cluster-based detector for point and group anomalies.

    Parameters
    ----------
    ...
    p_property : PropertyDefinition
        Cluster property to be observed.
    p_cls_anomaly : type
        Type of anomaly events to be raised.
    ...
    """

    C_NAME = 'Point and Group'
    C_PROPERTY_DEFINITIONS : PropertyDefinition = []
    
## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_property : PropertyDefinition,
                  p_group_anomaly_det : bool = True,
                  p_cls_point_anomaly : type = PointAnomaly,
                  p_cls_spatial_group_anomaly : type = SpatialGroupAnomaly,
                  p_cls_temporal_group_anomaly : type = TemporalGroupAnomaly,
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  p_anomaly_buffer_size : int = 100,
                  **p_kwargs ):
        
        self._cls_point_anomaly = p_cls_point_anomaly
        self._cls_group_anomaly = p_cls_group_anomaly
        self._cls_spatial_group_anomaly = p_cls_spatial_group_anomaly
        self._cls_temporal_group_anomaly = p_cls_temporal_group_anomaly
        self._group_anomaly_det = p_group_anomaly_det

        super().__init__( p_clusterer=p_clusterer,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging= p_logging,                        
                          **p_kwargs)


## -------------------------------------------------------------------------------------------------    
def _run(self, 
         p_inst: InstDict, 
         p_cluster : Cluster, 
         p_property : Property):

    # 1 Get all the clusters from the clusterer
    clusters = self._clusterer.get_clusters()

    # 2 Get the cluster property to be observed
    cluster_size : Property = getattr(p_cluster, p_property)

    # 3 Get average cluster size
    avg_cluster_size = sum([getattr(c, cprop_size)[0] for c in clusters]) / len(clusters)

    # 4 Calculater the threshold for the cluster size
    thres_size = avg_cluster_size * 0.05 

    # 5 Check for the  anomaly clusters
    if 1 <= cluster_size <= thres_size:
        # 5.1 Create a new spatial group anomaly 
        t_stamp = datetime.now()
        spatial_group_anomaly = self._cls_spatial_group_anomaly( p_clusters = {p_cluster.id : p_cluster},
                                                                 p_tstamp = t_stamp,
                                                                 p_visualize = self.get_visualize,
                                                                 p_raising_object = self)
        return spatial_group_anomaly

    elif cluster_size == 1:
        # 5.2 Create a new point anomaly 
        t_stamp = datetime.now()
        point_anomaly = self._cls_point_anomaly( p_clusters = {p_cluster.id : p_cluster},
                                                     p_tstamp = t_stamp,
                                                     p_visualize = self.get_visualize,
                                                     p_raising_object = self)
        return point_anomaly



    