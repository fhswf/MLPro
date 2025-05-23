## -- ----------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.tasks.anomalydetectors.cb_detectors
## -- Module  : point_group_anomaly_detector.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-03-28  0.0.0     DS       Creation
## -- 2025-04-03  0.0.1     DS/DA    Buffering of anomalies added.
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.1 (2025-04-03)

This module provides cluster based point and group anomaly detector algorithm.
"""

from mlpro.bf.various import Log
from mlpro.bf.math.properties import *
from mlpro.bf.streams import InstDict, InstTypeNew, Stream
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
    p_clusterer : ClusterAnalyzer
        The cluster analyzer to be used for detecting anomalies.
    p_property : PropertyDefinition
        Cluster property to be observed.
    p_group_anomaly_det : bool
        Whether to enable group anomaly detection.
    p_cls_point_anomaly : type
        Type of point anomaly events to be raised.
    p_cls_spatial_group_anomaly : type
        Type of spatial group anomaly events to be raised.
    p_cls_temporal_group_anomaly : type
        Type of temporal group anomaly events to be raised.
    p_name : str
        Name of the anomaly detector.
    p_range_max : int
        Maximum range for the task.
    p_ada : bool
        Whether to enable adaptive behavior.
    p_duplicate_data : bool
        Whether to allow duplicate data.
    p_visualize : bool
        Whether to enable visualization.
    p_logging : int
        Logging level.
    p_anomaly_buffer_size : int
        Size of the anomaly buffer.
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
        self._cb_anomalies = {}

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
         p_tstamp: float,
         p_cluster : Cluster, 
         p_property : Property):

        # 1 Get all the clusters from the clusterer
        clusters = self._clusterer.get_clusters()

        # 2 Get the cluster property to be observed
        cluster_size : Property = getattr(p_cluster, p_property)

        # 3 Get average cluster size
        avg_cluster_size = sum([getattr(c, cprop_size[0]).value for c in clusters]) / len(clusters)

        # 4 Calculater the threshold for the cluster size
        thres_size = avg_cluster_size * 0.05 

        # 5 Check for the  anomaly clusters
        if 1 < cluster_size <= thres_size:
            # 5.1 Create a new spatial group anomaly 
            try:
                cb_anomaly = self._cb_anomalies[p_cluster.id]

                create_anomaly = type(cb_anomaly) == self._cls_point_anomaly
                    
            except:

                create_anomaly = True

            if create_anomaly:
                # 5.1.1 Create a new spatial group anomaly

                spatial_group_anomaly = self._cls_spatial_group_anomaly( p_clusters = {p_cluster.id : p_cluster},
                                                                         p_tstamp = self.get_tstamp(),
                                                                         p_visualize = self.get_visualize,
                                                                         p_raising_object = self)
                                                            
                # 5.2 Raise an anomaly event
                self._raise_anomaly_event( p_anomaly = spatial_group_anomaly)

        elif cluster_size == 1:
            # 5.2 Create a new point anomaly 
            try:
                cb_anomaly = self._cb_anomalies[p_cluster.id]

                create_anomaly = (type(cb_anomaly) == self._cls_group_anomaly or type(cb_anomaly) == self._cls_spatial_group_anomaly)

            except:
                create_anomaly = True

            # t_stamp = datetime.now() # _get_tstamp()
            
            if create_anomaly:
                point_anomaly = self._cls_point_anomaly( p_clusters = {p_cluster.id : p_cluster},
                                                        p_tstamp = self.get_tstamp(),
                                                        p_visualize = self.get_visualize,
                                                        p_raising_object = self)

                self._raise_anomaly_event(p_anomaly = point_anomaly)


## -------------------------------------------------------------------------------------------------

    def _buffer_anomaly(self, p_anomaly:AnomalyCB):
        """
        Method to be used to add a new anomaly. Please use as part of your algorithm.

        Parameters
        ----------
        p_anomaly : AnomalyCB
            Anomaly object to be added.
        """
        super()._buffer_anomaly(p_anomaly)
        self._cb_anomalies[p_anomaly.clusters.values()[0].id] = p_anomaly

    