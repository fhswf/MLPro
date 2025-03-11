## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.driftdetectors.clusterbased.generic
## -- Module  : detector_point_group.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-11  0.1.0     DS/DA    Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2025-03-11)

This module provides an implementation of a generic cluster-based detector for point and group anomalies.
"""

from mlpro.bf.various import Log
from mlpro.bf.math.properties import *
from mlpro.oa.streams import OAStreamTask
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer, Cluster
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import cprop_size_prev
from mlpro.oa.streams.tasks.anomalydetectors.clusterbased.generic.basics import AnomalyDetectorCBGenSingle
from mlpro.oa.streams.tasks.anomalydetectors.anomalies.clusterbased import AnomalyCB, PointAnomaly, GroupAnomaly



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AnomalyDetectorCBGenPAGA ( AnomalyDetectorCBGenSingle ):
    """
    Implementation of a generic cluster-based detector for point and group anomalies.

    Parameters
    ----------
    ...
    p_property : PropertyDefinition
        Cluster property to be observed.
    p_cls_anomaly : type
        Type of anomaly events to be raised.
    ...
    """

    C_NAME = 'Point and group'

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_clusterer : ClusterAnalyzer,
                  p_property : PropertyDefinition = cprop_size_prev,
                  p_group_anomaly_det : bool = True,
                  p_cls_point_anomaly : type = PointAnomaly,
                  p_cls_group_anomaly : type = GroupAnomaly,
                  p_name : str = None,
                  p_range_max = OAStreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False,
                  p_logging=Log.C_LOG_ALL,
                  p_anomaly_buffer_size : int = 100,
                  **p_kwargs ):
        
        self._cls_group_anomaly = p_cls_group_anomaly
        self._group_anomaly_det = p_group_anomaly_det

        super().__init__( p_clusterer=p_clusterer,
                          p_property = p_property,
                          p_cls_anomaly = p_cls_point_anomaly,
                          p_name = p_name,
                          p_range_max = p_range_max,
                          p_ada = p_ada,
                          p_duplicate_data = p_duplicate_data,
                          p_visualize = p_visualize,
                          p_logging= p_logging,
                          p_anomaly_buffer_size = p_anomaly_buffer_size,
                          **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def _detect_anomaly( self, 
                         p_cluster : Cluster, 
                         p_properties : PropertyDefinitions, 
                         **p_kwargs ) -> AnomalyCB:
        """
        ...
        """

        # 1 Locate cluster property to be examined
        cluster_size : Property = getattr(p_cluster, p_properties[0][0])
        
        # 2 Check whether the cluster size 1 remains for at least 
        if cluster_size.value != 1: return None
        if ( cluster_size.value_prev is None ) or ( cluster_size.value_prev != 1 ): return None

        # 3 Create a new point anomaly object
        return self._cls_anomaly( p_clusters = { p_cluster.id : p_cluster },
                                  # p_properties : dict = ,
                                  # p_tstamp : datetime = None,
                                  p_visualize : bool = self.get_visualization(),
                                  p_raising_object : object = self )